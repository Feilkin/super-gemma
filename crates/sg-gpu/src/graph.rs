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
/// reuse across dispatches.
pub struct GraphRecorder<'a> {
    ctx: &'a GpuContext,
    builder: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
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
        let mut ticks = vec![0u64; self.count as usize];
        self.pool
            .get_results(0..self.count, &mut ticks, QueryResultFlags::WAIT)
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
        let mut recorder = GraphRecorder { ctx: self, builder };
        record(&mut recorder)?;
        let cb = recorder.builder.build().map_err(GpuError::validated)?;
        Ok(CommandGraph { cb })
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
