//! One-shot blocking dispatch, used by kernel parity tests and microbenches.
//! The production decode/prefill path uses pre-recorded command graphs
//! instead (plan 02 §submission model; M2 step 8).

use bytemuck::Pod;

use crate::{BufferBinding, GpuContext, GpuError, Kernel};

impl GpuContext {
    /// Bind `writes` as descriptor set 0, push `push` (if any), dispatch
    /// `groups`, submit, and wait for completion.
    pub fn dispatch_blocking<P: Pod>(
        &self,
        kernel: &Kernel,
        writes: Vec<BufferBinding>,
        push: Option<P>,
        groups: [u32; 3],
    ) -> Result<(), GpuError> {
        let mut rec = self.begin_recorder(true)?;
        // On error `rec` drops, freeing its command/descriptor pools.
        rec.dispatch(kernel, writes, push, groups)?;
        self.submit_recorder_blocking(rec)
    }
}
