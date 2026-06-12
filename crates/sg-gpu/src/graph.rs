//! Pre-recorded command graphs (plan 02 §submission model): the decode step
//! and prefill chunk are recorded ONCE and re-submitted every token/chunk.
//! Per-step dynamic state (position, KV lengths, prefill history offset)
//! lives in a small step buffer the CPU rewrites between submits — one
//! 16-byte write instead of re-recording; everything else (pipelines,
//! descriptor sets, push constants, dispatch grids) is baked at record
//! time. Split-K attention grids are recorded at a fixed split count and
//! shrink via the step buffer's kv_len (empty splits are handled by the
//! reducers).
//!
//! Also here: [`GpuTimer`], timestamp queries recorded into a graph for
//! per-kernel GPU timing (plan 02 §performance, plan 06 benchmarks).

use std::sync::Arc;

use vulkano::buffer::{BufferContents, BufferUsage, Subbuffer};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer,
    PrimaryCommandBufferAbstract,
};
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::PipelineBindPoint;
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::{GpuFuture, PipelineStage};

use crate::{GpuContext, GpuError, Kernel};

/// Per-step dynamic state read by kernels from the step buffer (binding
/// layout documented in each shader). The engine rewrites this between
/// submits of a pre-recorded graph; field semantics are pinned here and in
/// the shaders' comments.
#[derive(Debug, Clone, Copy, Default)]
pub struct StepState {
    /// Absolute position of the first token appended/decoded this step
    /// (`kv_append_*` destination).
    pub pos: u32,
    /// Valid slots in the sliding ring (`attn_decode_sliding`).
    pub kv_len_sliding: u32,
    /// Tokens in the resident global KV (`attn_decode_global`).
    pub kv_len_global: u32,
    /// Prefill history length: query token i sits at key index q0 + i
    /// (`attn_prefill_*`).
    pub q0: u32,
}

/// Number of u32 words in a step buffer.
pub const STEP_WORDS: u64 = 4;

impl StepState {
    /// Write into a step buffer between submits. Fails if the buffer is
    /// still in use by an unfinished submission.
    pub fn write_to(&self, buf: &Subbuffer<[u32]>) -> Result<(), GpuError> {
        let mut w = buf
            .write()
            .map_err(|e| GpuError::Validation(e.to_string()))?;
        w[0] = self.pos;
        w[1] = self.kv_len_sliding;
        w[2] = self.kv_len_global;
        w[3] = self.q0;
        Ok(())
    }
}

/// A command buffer recorded once and submitted many times.
pub struct CommandGraph {
    cb: Arc<PrimaryAutoCommandBuffer>,
}

/// Records dispatches (and optional timestamps) into a [`CommandGraph`].
/// vulkano's auto-sync inserts the pipeline barriers implied by buffer
/// reuse across dispatches — **as far as it can see**: usage is derived
/// from SPIR-V reflection, which misses cooperative-matrix accesses (the
/// same blindness `kernel.rs` works around for descriptor layouts). A
/// buffer consumed only via coopLoad gets NO barrier after its producer;
/// record a `touch` dispatch on it first (see `shaders/touch.wgsl`).
pub struct GraphRecorder<'a> {
    ctx: &'a GpuContext,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    /// When profiling: the timer plus one label per recorded interval.
    /// `auto` stamps after every dispatch (`record_graph_profiled` — keep
    /// such graphs SMALL: per-dispatch timestamps at graph scale melted
    /// both vulkano's recording and the GPU ring, see sg-bench/profile.rs);
    /// otherwise only explicit [`Self::mark`] calls stamp.
    prof: Option<Prof<'a>>,
}

struct Prof<'a> {
    timer: &'a GpuTimer,
    labels: Vec<&'static str>,
    auto: bool,
}

impl GraphRecorder<'_> {
    /// Record one dispatch: bind `kernel`, bind `writes` as set 0, push
    /// `push` (if any), dispatch `groups`.
    pub fn dispatch<P: BufferContents>(
        &mut self,
        kernel: &Kernel,
        writes: Vec<WriteDescriptorSet>,
        push: Option<P>,
        groups: [u32; 3],
    ) -> Result<&mut Self, GpuError> {
        let layout = kernel.layout().clone();
        let set = DescriptorSet::new(
            self.ctx.descriptor_set_allocator().clone(),
            layout.set_layouts()[0].clone(),
            writes,
            [],
        )
        .map_err(GpuError::validated)?;
        self.builder
            .bind_pipeline_compute(kernel.pipeline().clone())
            .map_err(|e| GpuError::Pipeline(e.to_string()))?
            .bind_descriptor_sets(PipelineBindPoint::Compute, layout.clone(), 0, set)
            .map_err(|e| GpuError::Pipeline(e.to_string()))?;
        if let Some(push) = push {
            self.builder
                .push_constants(layout, 0, push)
                .map_err(|e| GpuError::Pipeline(e.to_string()))?;
        }
        // SAFETY: dispatch bounds are the caller's contract with the kernel;
        // all our kernels bounds-check against arrayLength or baked sizes.
        unsafe { self.builder.dispatch(groups) }.map_err(|e| GpuError::Pipeline(e.to_string()))?;

        if self.prof.as_ref().is_some_and(|p| p.auto) {
            let name = kernel.name;
            self.mark(name)?;
        }
        Ok(self)
    }

    /// Record a profiling timestamp closing an interval labelled `label`.
    /// Only valid inside a profiled recording.
    pub fn mark(&mut self, label: &'static str) -> Result<&mut Self, GpuError> {
        let Some(prof) = self.prof.take() else {
            return Err(GpuError::Validation(
                "mark() outside a profiled recording".into(),
            ));
        };
        let index = prof.labels.len() as u32 + 1;
        if index >= prof.timer.count {
            return Err(GpuError::Validation(format!(
                "profiling timer too small: {} queries for >{index} timestamps",
                prof.timer.count
            )));
        }
        self.timestamp(prof.timer, index)?;
        let mut prof = prof;
        prof.labels.push(label);
        self.prof = Some(prof);
        Ok(self)
    }

    /// Reset a timer's queries; must be recorded before its `timestamp`s.
    pub fn reset_timer(&mut self, timer: &GpuTimer) -> Result<&mut Self, GpuError> {
        // SAFETY: the pool is not in use by an earlier submission — graphs
        // are submitted blocking, and recording happens before any submit.
        unsafe {
            self.builder
                .reset_query_pool(timer.pool.clone(), 0..timer.count)
        }
        .map_err(|e| GpuError::Pipeline(e.to_string()))?;
        Ok(self)
    }

    /// Record a timestamp: query `index` captures when all previously
    /// recorded work has completed.
    pub fn timestamp(&mut self, timer: &GpuTimer, index: u32) -> Result<&mut Self, GpuError> {
        // SAFETY: query index validity is checked by vulkano; BottomOfPipe
        // orders the write after all prior commands.
        unsafe {
            self.builder
                .write_timestamp(timer.pool.clone(), index, PipelineStage::BottomOfPipe)
        }
        .map_err(|e| GpuError::Pipeline(e.to_string()))?;
        Ok(self)
    }
}

/// GPU timestamp queries for per-kernel timing inside a graph.
pub struct GpuTimer {
    pool: Arc<QueryPool>,
    /// Nanoseconds per timestamp tick (device property).
    period_ns: f64,
    count: u32,
}

impl GpuTimer {
    /// Read all timestamps in nanoseconds (waits for availability). Only
    /// meaningful after the graph containing the writes has been submitted.
    pub fn read_ns(&self) -> Result<Vec<f64>, GpuError> {
        self.read_ns_prefix(self.count)
    }

    /// Read the first `n` timestamps — for profiled graphs that wrote
    /// fewer queries than the pool holds (waiting on unwritten queries
    /// would block forever).
    pub fn read_ns_prefix(&self, n: u32) -> Result<Vec<f64>, GpuError> {
        let mut ticks = vec![0u64; n as usize];
        self.pool
            .get_results(0..n, &mut ticks, QueryResultFlags::WAIT)
            .map_err(GpuError::validated)?;
        Ok(ticks
            .into_iter()
            .map(|t| t as f64 * self.period_ns)
            .collect())
    }
}

impl GpuContext {
    /// A step buffer for [`StepState`] (storage, host-writable).
    pub fn new_step_buffer(&self) -> Result<Subbuffer<[u32]>, GpuError> {
        self.new_buffer::<u32>(STEP_WORDS, BufferUsage::STORAGE_BUFFER)
    }

    /// A timestamp pool with `count` queries.
    pub fn new_timer(&self, count: u32) -> Result<GpuTimer, GpuError> {
        let pool = QueryPool::new(
            self.device().clone(),
            QueryPoolCreateInfo {
                query_count: count,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .map_err(GpuError::validated)?;
        let period_ns = self
            .device()
            .physical_device()
            .properties()
            .timestamp_period as f64;
        Ok(GpuTimer {
            pool,
            period_ns,
            count,
        })
    }

    /// Record a graph once; submit it many times with [`Self::submit_blocking`].
    pub fn record_graph<F>(&self, record: F) -> Result<CommandGraph, GpuError>
    where
        F: FnOnce(&mut GraphRecorder<'_>) -> Result<(), GpuError>,
    {
        let builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator().clone(),
            self.queue().queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .map_err(GpuError::validated)?;
        let mut recorder = GraphRecorder {
            ctx: self,
            builder,
            prof: None,
        };
        record(&mut recorder)?;
        let cb = recorder.builder.build().map_err(GpuError::validated)?;
        Ok(CommandGraph { cb })
    }

    /// Like [`Self::record_graph`], but every dispatch is followed by a
    /// timestamp into `timer` (query 0 marks the start). Returns one label
    /// per interval: dispatch `i`'s duration is `ts[i+1] − ts[i]`.
    ///
    /// **Keep these graphs small** (≲ 100 dispatches): a full decode graph
    /// (~1400 dispatches) with per-dispatch timestamps took minutes of
    /// auto-sync recording CPU and then tripped the amdgpu ring watchdog.
    /// For graph-scale profiling use [`Self::record_graph_with_marks`].
    ///
    /// Attribution caveat: BottomOfPipe timestamps partition the total
    /// time exactly, but where adjacent dispatches overlap (no barrier
    /// between them), an interval's time may include a neighbour's tail —
    /// fine for ranking, not for microbenchmarking a single kernel.
    pub fn record_graph_profiled<F>(
        &self,
        timer: &GpuTimer,
        record: F,
    ) -> Result<(CommandGraph, Vec<&'static str>), GpuError>
    where
        F: FnOnce(&mut GraphRecorder<'_>) -> Result<(), GpuError>,
    {
        self.record_profiled_inner(timer, true, record)
    }

    /// Profiled recording with MANUAL interval boundaries: only
    /// [`GraphRecorder::mark`] calls stamp (query 0 = start). The cheap
    /// way to profile big graphs — e.g. one mark per layer.
    pub fn record_graph_with_marks<F>(
        &self,
        timer: &GpuTimer,
        record: F,
    ) -> Result<(CommandGraph, Vec<&'static str>), GpuError>
    where
        F: FnOnce(&mut GraphRecorder<'_>) -> Result<(), GpuError>,
    {
        self.record_profiled_inner(timer, false, record)
    }

    fn record_profiled_inner<F>(
        &self,
        timer: &GpuTimer,
        auto: bool,
        record: F,
    ) -> Result<(CommandGraph, Vec<&'static str>), GpuError>
    where
        F: FnOnce(&mut GraphRecorder<'_>) -> Result<(), GpuError>,
    {
        let builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator().clone(),
            self.queue().queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .map_err(GpuError::validated)?;
        let mut recorder = GraphRecorder {
            ctx: self,
            builder,
            prof: None,
        };
        recorder.reset_timer(timer)?;
        recorder.timestamp(timer, 0)?;
        recorder.prof = Some(Prof {
            timer,
            labels: Vec::new(),
            auto,
        });
        record(&mut recorder)?;
        let labels = recorder.prof.take().expect("prof attached above").labels;
        let cb = recorder.builder.build().map_err(GpuError::validated)?;
        Ok((CommandGraph { cb }, labels))
    }

    /// Submit a pre-recorded graph and wait for completion.
    pub fn submit_blocking(&self, graph: &CommandGraph) -> Result<(), GpuError> {
        graph
            .cb
            .clone()
            .execute(self.queue().clone())
            .map_err(|e| GpuError::Pipeline(e.to_string()))?
            .then_signal_fence_and_flush()
            .map_err(GpuError::validated)?
            .wait(None)
            .map_err(GpuError::validated)?;
        Ok(())
    }
}
