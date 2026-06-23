//! Pre-recorded command graphs (plan 02 §submission model): the decode step
//! and prefill chunk are recorded ONCE and re-submitted every token/chunk.
//! Per-step dynamic state (position, KV lengths, prefill history offset)
//! lives in a small step buffer the CPU rewrites between submits — one
//! 16-byte write instead of re-recording; everything else (pipelines,
//! descriptor sets, push constants, dispatch grids) is baked at record time.
//!
//! Barriers: on `ash` we insert the pipeline barriers ourselves. [`GraphRecorder::dispatch`]
//! emits a compute→compute global memory barrier **before every dispatch after
//! the first**, so a GEMM's `coopStore` to `y` is ordered before the next
//! dispatch's `coopLoad` of it — the dependency vulkano's auto-sync could not
//! see (it reflects SPIR-V buffer usage, which misses cooperative-matrix
//! accesses). Independent dispatches that could legitimately overlap can use
//! [`GraphRecorder::dispatch_overlapping`] (no preceding barrier).
//!
//! Also here: [`GpuTimer`], timestamp queries recorded into a graph for
//! per-kernel GPU timing (plan 02 §performance, plan 06 benchmarks).

use std::ffi::CString;
use std::sync::Arc;

use ash::vk;
use bytemuck::Pod;

use crate::buffer::{Buffer, BufferBinding, BufferInner, BufferUsage};
use crate::context::DeviceCtx;
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
    pub fn write_to(&self, buf: &Buffer<u32>) -> Result<(), GpuError> {
        let mut w = buf.write()?;
        w[0] = self.pos;
        w[1] = self.kv_len_sliding;
        w[2] = self.kv_len_global;
        w[3] = self.q0;
        Ok(())
    }
}

/// Grows descriptor pools on demand, one set per recorded dispatch. The pools
/// live as long as the [`CommandGraph`] (or [`GraphRecorder`]) that owns the
/// arena, so the sets recorded into a re-submittable graph stay valid.
struct DescriptorArena {
    ctx: Arc<DeviceCtx>,
    pools: Vec<vk::DescriptorPool>,
    remaining: u32,
}

/// Sets allocated per descriptor pool before a new pool is created.
const SETS_PER_POOL: u32 = 256;
/// Storage descriptors reserved per set (max bindings any kernel uses is well
/// under this).
const DESCRIPTORS_PER_SET: u32 = 16;

impl DescriptorArena {
    fn new(ctx: Arc<DeviceCtx>) -> Self {
        Self {
            ctx,
            pools: Vec::new(),
            remaining: 0,
        }
    }

    fn allocate(&mut self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, GpuError> {
        if self.remaining == 0 {
            self.add_pool()?;
        }
        let pool = *self.pools.last().expect("pool just ensured");
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);
        // SAFETY: valid device + pool; the pool has a free set (remaining > 0).
        let set = unsafe { self.ctx.device.allocate_descriptor_sets(&info) }
            .map_err(|e| GpuError::Vk(format!("allocate_descriptor_sets: {e}")))?[0];
        self.remaining -= 1;
        Ok(set)
    }

    fn add_pool(&mut self) -> Result<(), GpuError> {
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(SETS_PER_POOL * DESCRIPTORS_PER_SET)];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(SETS_PER_POOL)
            .pool_sizes(&sizes);
        // SAFETY: valid device; create info borrows live locals.
        let pool = unsafe { self.ctx.device.create_descriptor_pool(&info, None) }
            .map_err(|e| GpuError::Vk(format!("create_descriptor_pool: {e}")))?;
        self.pools.push(pool);
        self.remaining = SETS_PER_POOL;
        Ok(())
    }
}

impl Drop for DescriptorArena {
    fn drop(&mut self) {
        // SAFETY: the owning graph is no longer submitted (submits fence-wait),
        // so the sets are idle; destroying a pool frees all its sets.
        unsafe {
            for &pool in &self.pools {
                self.ctx.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

/// A command buffer recorded once and submitted many times.
pub struct CommandGraph {
    ctx: Arc<DeviceCtx>,
    command_pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    /// Descriptor pools backing the recorded sets — kept alive with the graph.
    _arena: DescriptorArena,
    /// Buffers bound into the graph — kept alive so their handles stay valid.
    _keep: Vec<Arc<BufferInner>>,
}

impl Drop for CommandGraph {
    fn drop(&mut self) {
        // SAFETY: submits are fence-waited, so no submission references this CB
        // when the graph is dropped; destroying the pool frees the CB. The
        // arena's pools drop after (field order), the kept buffers after that.
        unsafe {
            self.ctx.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

/// Records dispatches (and optional timestamps) into a [`CommandGraph`]. Owns
/// the in-progress command buffer and the descriptor arena until the graph is
/// built; on a recording error the resources are torn down (no leak).
pub struct GraphRecorder<'a> {
    ctx: &'a GpuContext,
    command_pool: vk::CommandPool,
    cb: vk::CommandBuffer,
    /// `Some` until ownership is moved into a [`CommandGraph`]; lets the `Drop`
    /// below free the descriptor pools on the error/blocking paths.
    arena: Option<DescriptorArena>,
    keep: Vec<Arc<BufferInner>>,
    /// Dispatches recorded so far — the auto-barrier is inserted before every
    /// dispatch except the first.
    dispatched: u32,
    /// Set once the resources have been handed to a `CommandGraph`, so `Drop`
    /// does not double-free the command pool.
    finished: bool,
    /// When profiling: the timer plus one label per recorded interval.
    /// `auto` stamps after every dispatch (`record_graph_profiled` — keep
    /// such graphs SMALL); otherwise only explicit [`Self::mark`] calls stamp.
    prof: Option<Prof<'a>>,
}

impl Drop for GraphRecorder<'_> {
    fn drop(&mut self) {
        // Cleans up an abandoned recording (closure errored, or a one-shot
        // blocking dispatch finished). If `finished`, a `CommandGraph` owns the
        // command pool now, so leave it. The `arena` Option drops its pools.
        if !self.finished {
            // SAFETY: nothing references this CB — the closure errored before
            // submit, or a blocking submit has already fence-waited.
            unsafe {
                self.ctx
                    .device()
                    .destroy_command_pool(self.command_pool, None);
            }
        }
    }
}

struct Prof<'a> {
    timer: &'a GpuTimer,
    labels: Vec<&'static str>,
    auto: bool,
}

impl GraphRecorder<'_> {
    /// Record one dispatch, preceded by the default compute→compute barrier
    /// (except for the first dispatch): bind `kernel`, bind `writes` as set 0,
    /// push `push` (if any), dispatch `groups`.
    pub fn dispatch<P: Pod>(
        &mut self,
        kernel: &Kernel,
        writes: Vec<BufferBinding>,
        push: Option<P>,
        groups: [u32; 3],
    ) -> Result<&mut Self, GpuError> {
        self.record_dispatch(kernel, writes, push, groups, true)
    }

    /// Like [`Self::dispatch`] but with NO preceding barrier — for independent
    /// dispatches that may legitimately overlap (concurrency experiments).
    pub fn dispatch_overlapping<P: Pod>(
        &mut self,
        kernel: &Kernel,
        writes: Vec<BufferBinding>,
        push: Option<P>,
        groups: [u32; 3],
    ) -> Result<&mut Self, GpuError> {
        self.record_dispatch(kernel, writes, push, groups, false)
    }

    fn record_dispatch<P: Pod>(
        &mut self,
        kernel: &Kernel,
        writes: Vec<BufferBinding>,
        push: Option<P>,
        groups: [u32; 3],
        auto_barrier: bool,
    ) -> Result<&mut Self, GpuError> {
        let device = self.ctx.device();
        let cb = self.cb;

        // Allocate + fill a descriptor set for this dispatch's bindings.
        let set = self
            .arena
            .as_mut()
            .expect("arena present while recording")
            .allocate(kernel.set_layout)?;
        let infos: Vec<vk::DescriptorBufferInfo> = writes
            .iter()
            .map(|w| {
                vk::DescriptorBufferInfo::default()
                    .buffer(w.keep.buffer)
                    .offset(w.offset)
                    .range(w.range)
            })
            .collect();
        let set_writes: Vec<vk::WriteDescriptorSet> = writes
            .iter()
            .enumerate()
            .map(|(i, w)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(w.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&infos[i]))
            })
            .collect();
        // SAFETY: valid device; writes/infos live through the call.
        unsafe { device.update_descriptor_sets(&set_writes, &[]) };
        // Keep the buffers alive for the graph's life.
        for w in writes {
            self.keep.push(w.keep);
        }

        // The migration's payoff: order the previous dispatch's writes before
        // this one's reads/writes.
        if auto_barrier && self.dispatched > 0 {
            emit_compute_barrier(device, cb);
        }

        // Name the dispatch for RGP/SQTT: RADV records the label region as a
        // marker so each event in a capture shows its kernel. Inert on the GPU.
        let label_name = self
            .ctx
            .ctx
            .debug_utils
            .as_ref()
            .map(|_| CString::new(kernel.name).expect("kernel name has no NUL"));
        if let (Some(du), Some(name)) = (&self.ctx.ctx.debug_utils, &label_name) {
            let label = vk::DebugUtilsLabelEXT::default().label_name(name);
            // SAFETY: balanced begin/end around this one dispatch.
            unsafe { du.cmd_begin_debug_utils_label(cb, &label) };
        }

        // SAFETY: valid CB + pipeline/layout/set; dispatch bounds are the
        // caller's contract (kernels bounds-check against arrayLength/sizes).
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, kernel.pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[set],
                &[],
            );
            if let Some(push) = push.as_ref() {
                device.cmd_push_constants(
                    cb,
                    kernel.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytemuck::bytes_of(push),
                );
            }
            device.cmd_dispatch(cb, groups[0], groups[1], groups[2]);
        }

        if let Some(du) = &self.ctx.ctx.debug_utils {
            // SAFETY: closes the region begun above (balanced).
            unsafe { du.cmd_end_debug_utils_label(cb) };
        }

        self.dispatched += 1;
        if self.prof.as_ref().is_some_and(|p| p.auto) {
            let name = kernel.name;
            self.mark(name)?;
        }
        Ok(self)
    }

    /// Record an explicit compute→compute global memory barrier.
    pub fn barrier(&mut self) -> &mut Self {
        emit_compute_barrier(self.ctx.device(), self.cb);
        self
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
        // SAFETY: valid CB + pool; recorded before any submit, so the pool is
        // not in use by an earlier submission.
        unsafe {
            self.ctx
                .device()
                .cmd_reset_query_pool(self.cb, timer.pool, 0, timer.count);
        }
        Ok(self)
    }

    /// Record a timestamp: query `index` captures when all previously recorded
    /// work has completed (bottom-of-pipe).
    pub fn timestamp(&mut self, timer: &GpuTimer, index: u32) -> Result<&mut Self, GpuError> {
        // SAFETY: valid CB + pool; BOTTOM_OF_PIPE orders the write after all
        // prior commands; index < count is the caller's contract (checked in mark).
        unsafe {
            self.ctx.device().cmd_write_timestamp(
                self.cb,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                timer.pool,
                index,
            );
        }
        Ok(self)
    }
}

/// Emit a global compute→compute memory barrier (shader writes visible to
/// later shader reads/writes).
fn emit_compute_barrier(device: &ash::Device, cb: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    // SAFETY: valid CB; the barrier is a value passed by slice.
    unsafe {
        device.cmd_pipeline_barrier(
            cb,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[barrier],
            &[],
            &[],
        );
    }
}

/// GPU timestamp queries for per-kernel timing inside a graph.
pub struct GpuTimer {
    ctx: Arc<DeviceCtx>,
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick (device property).
    period_ns: f64,
    count: u32,
}

impl GpuTimer {
    /// Number of queries in the pool.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Read all timestamps in nanoseconds (waits for availability). Only
    /// meaningful after the graph containing the writes has been submitted.
    pub fn read_ns(&self) -> Result<Vec<f64>, GpuError> {
        self.read_ns_prefix(self.count)
    }

    /// Read the first `n` timestamps — for profiled graphs that wrote fewer
    /// queries than the pool holds (waiting on unwritten queries would block).
    pub fn read_ns_prefix(&self, n: u32) -> Result<Vec<f64>, GpuError> {
        let mut ticks = vec![0u64; n as usize];
        // SAFETY: valid pool; `ticks` has `n` u64 slots; TYPE_64 matches u64.
        unsafe {
            self.ctx.device.get_query_pool_results(
                self.pool,
                0,
                &mut ticks,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            )
        }
        .map_err(|e| GpuError::Vk(format!("get_query_pool_results: {e}")))?;
        Ok(ticks
            .into_iter()
            .map(|t| t as f64 * self.period_ns)
            .collect())
    }
}

impl Drop for GpuTimer {
    fn drop(&mut self) {
        // SAFETY: timer holds an Arc<DeviceCtx>; the pool is idle (results are
        // read with WAIT, submits fence-wait) when the last timer is dropped.
        unsafe { self.ctx.device.destroy_query_pool(self.pool, None) };
    }
}

impl GpuContext {
    /// A step buffer for [`StepState`] (storage, host-writable).
    pub fn new_step_buffer(&self) -> Result<Buffer<u32>, GpuError> {
        self.new_buffer::<u32>(STEP_WORDS, BufferUsage::STORAGE_BUFFER)
    }

    /// A timestamp pool with `count` queries.
    pub fn new_timer(&self, count: u32) -> Result<GpuTimer, GpuError> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(count);
        // SAFETY: valid device; create info borrows nothing past the call.
        let pool = unsafe { self.device().create_query_pool(&info, None) }
            .map_err(|e| GpuError::Vk(format!("create_query_pool: {e}")))?;
        Ok(GpuTimer {
            ctx: self.ctx.clone(),
            pool,
            period_ns: self.ctx.timestamp_period as f64,
            count,
        })
    }

    /// Begin recording: create a command pool + primary command buffer and put
    /// it in the recording state. `one_time` chooses the begin-usage flag.
    pub(crate) fn begin_recorder(&self, one_time: bool) -> Result<GraphRecorder<'_>, GpuError> {
        let device = self.device();
        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(self.ctx.queue_family)
            .flags(vk::CommandPoolCreateFlags::empty());
        // SAFETY: valid device; create info borrows nothing past the call.
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| GpuError::Vk(format!("create_command_pool: {e}")))?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: valid device + pool just created.
        let cb = match unsafe { device.allocate_command_buffers(&alloc) } {
            Ok(v) => v[0],
            Err(e) => {
                // SAFETY: destroy the pool on the failure path.
                unsafe { device.destroy_command_pool(command_pool, None) };
                return Err(GpuError::Vk(format!("allocate_command_buffers: {e}")));
            }
        };
        let flags = if one_time {
            vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
        } else {
            vk::CommandBufferUsageFlags::empty()
        };
        let begin = vk::CommandBufferBeginInfo::default().flags(flags);
        // SAFETY: fresh CB in the initial state.
        if let Err(e) = unsafe { device.begin_command_buffer(cb, &begin) } {
            // SAFETY: destroy the pool on the failure path.
            unsafe { device.destroy_command_pool(command_pool, None) };
            return Err(GpuError::Vk(format!("begin_command_buffer: {e}")));
        }
        Ok(GraphRecorder {
            ctx: self,
            command_pool,
            cb,
            arena: Some(DescriptorArena::new(self.ctx.clone())),
            keep: Vec::new(),
            dispatched: 0,
            finished: false,
            prof: None,
        })
    }

    /// Record a graph once; submit it many times with [`Self::submit_blocking`].
    pub fn record_graph<F>(&self, record: F) -> Result<CommandGraph, GpuError>
    where
        F: FnOnce(&mut GraphRecorder<'_>) -> Result<(), GpuError>,
    {
        self.record_inner(record).map(|(g, _)| g)
    }

    /// Like [`Self::record_graph`], but every dispatch is followed by a
    /// timestamp into `timer` (query 0 marks the start). Returns one label
    /// per interval: dispatch `i`'s duration is `ts[i+1] − ts[i]`.
    ///
    /// **Keep these graphs small** (≲ 100 dispatches): a full decode graph
    /// (~1400 dispatches) with per-dispatch timestamps tripped the amdgpu ring
    /// watchdog. For graph-scale profiling use [`Self::record_graph_with_marks`].
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
    /// [`GraphRecorder::mark`] calls stamp (query 0 = start). The cheap way to
    /// profile big graphs — e.g. one mark per layer.
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

    /// Drive the closure, then finish into a `CommandGraph`. On any error `rec`
    /// drops, freeing the half-recorded command pool and descriptor pools.
    fn record_inner<F>(&self, record: F) -> Result<(CommandGraph, Vec<&'static str>), GpuError>
    where
        F: FnOnce(&mut GraphRecorder<'_>) -> Result<(), GpuError>,
    {
        let mut rec = self.begin_recorder(false)?;
        record(&mut rec)?;
        self.finish_recorder(rec)
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
        let mut rec = self.begin_recorder(false)?;
        let run = (|| {
            rec.reset_timer(timer)?;
            rec.timestamp(timer, 0)?;
            rec.prof = Some(Prof {
                timer,
                labels: Vec::new(),
                auto,
            });
            record(&mut rec)
        })();
        // On error `rec` drops, freeing the half-recorded pools.
        run?;
        self.finish_recorder(rec)
    }

    /// End the command buffer and move the recorder's resources into a graph.
    fn finish_recorder(
        &self,
        mut rec: GraphRecorder<'_>,
    ) -> Result<(CommandGraph, Vec<&'static str>), GpuError> {
        let labels = rec.prof.take().map(|p| p.labels).unwrap_or_default();
        // SAFETY: valid CB in the recording state. On error, `rec`'s Drop frees
        // the command pool (finished is still false).
        unsafe { self.device().end_command_buffer(rec.cb) }
            .map_err(|e| GpuError::Vk(format!("end_command_buffer: {e}")))?;
        // Hand the resources to the graph. Mark finished so `rec`'s Drop leaves
        // the command pool alone; `take` the owning fields (Drop forbids a move
        // out of `rec`).
        rec.finished = true;
        let arena = rec.arena.take().expect("arena present at finish");
        let keep = std::mem::take(&mut rec.keep);
        let graph = CommandGraph {
            ctx: self.ctx.clone(),
            command_pool: rec.command_pool,
            cb: rec.cb,
            _arena: arena,
            _keep: keep,
        };
        Ok((graph, labels))
    }

    /// End a one-shot recording, submit it, block until done, then let `rec`'s
    /// Drop free the command pool. Used by `dispatch_blocking`.
    pub(crate) fn submit_recorder_blocking(
        &self,
        rec: GraphRecorder<'_>,
    ) -> Result<(), GpuError> {
        // SAFETY: valid CB in the recording state.
        unsafe { self.device().end_command_buffer(rec.cb) }
            .map_err(|e| GpuError::Vk(format!("end_command_buffer: {e}")))?;
        let result = self.submit_and_wait(rec.cb);
        // `rec` drops here: command pool + descriptor pools freed after the
        // fence wait above guaranteed the submission completed.
        result
    }

    /// Submit a pre-recorded graph and wait for completion.
    pub fn submit_blocking(&self, graph: &CommandGraph) -> Result<(), GpuError> {
        self.submit_and_wait(graph.cb)
    }

    /// Submit one command buffer and block on a fence until it completes.
    pub(crate) fn submit_and_wait(&self, cb: vk::CommandBuffer) -> Result<(), GpuError> {
        let device = self.device();
        let fence_ci = vk::FenceCreateInfo::default();
        // SAFETY: valid device.
        let fence = unsafe { device.create_fence(&fence_ci, None) }
            .map_err(|e| GpuError::Vk(format!("create_fence: {e}")))?;
        let cbs = [cb];
        let submit = vk::SubmitInfo::default().command_buffers(&cbs);
        // SAFETY: valid queue + CB; the fence is signalled on completion.
        let result = unsafe {
            device
                .queue_submit(self.ctx.queue, &[submit], fence)
                .map_err(|e| GpuError::Vk(format!("queue_submit: {e}")))
                .and_then(|()| {
                    device
                        .wait_for_fences(&[fence], true, u64::MAX)
                        .map_err(|e| GpuError::Vk(format!("wait_for_fences: {e}")))
                })
        };
        // SAFETY: fence no longer referenced after the wait returns.
        unsafe { device.destroy_fence(fence, None) };
        result
    }
}
