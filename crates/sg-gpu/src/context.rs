//! Device and queue setup: one instance, one device, one compute queue, with
//! the features the plans assert (plan 02 — all confirmed present on the
//! target box by the M0 probe).

use std::sync::Arc;

use vulkano::VulkanLibrary;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, DeviceFeatures, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::memory::allocator::StandardMemoryAllocator;

use crate::GpuError;

/// The device features every kernel relies on; absence is a startup error,
/// not a fallback (this server targets exactly one machine).
fn required_features() -> DeviceFeatures {
    DeviceFeatures {
        shader_float16: true,
        storage_buffer16_bit_access: true,
        uniform_and_storage_buffer16_bit_access: true,
        timeline_semaphore: true,
        shader_int8: true,
        // int8 coopmat operands load from array<i8> storage buffers.
        storage_buffer8_bit_access: true,
        ..Default::default()
    }
}

/// The Vulkan compute context: one device, one compute queue (the dedicated
/// GPU thread in plan 00 owns the `Queue`; nothing else submits).
pub struct GpuContext {
    device: Arc<Device>,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    /// Default subgroup size reported by the driver (64 on the target's RDNA
    /// 3.5 per the M0 probe). Kernel variants bake this in via defines.
    pub subgroup_size: u32,
    /// Whether `VK_KHR_cooperative_matrix` was enabled (true on target;
    /// coopmat GEMM variants require it).
    pub cooperative_matrix: bool,
    /// Whether `VK_EXT_debug_utils` was enabled — gates per-dispatch label
    /// regions (kernel names in RGP/SQTT captures).
    pub debug_utils: bool,
}

impl GpuContext {
    /// Create the context on the best available device. Errors (rather than
    /// panics) when no Vulkan or no capable GPU is present, so off-target
    /// tests can skip gracefully.
    pub fn new() -> Result<Self, GpuError> {
        let library = VulkanLibrary::new()?;
        // `ext_debug_utils` lets us name each dispatch with a command-buffer
        // label region (`GraphRecorder::dispatch`); RADV emits those as SQTT
        // markers so RGP captures show which kernel each event is. Optional —
        // absence just means unlabelled traces, never a failure.
        let debug_utils = library.supported_extensions().ext_debug_utils;
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: InstanceExtensions {
                    ext_debug_utils: debug_utils,
                    ..InstanceExtensions::empty()
                },
                ..Default::default()
            },
        )
        .map_err(GpuError::validated)?;

        let required = required_features();
        let mut candidates: Vec<Arc<PhysicalDevice>> = instance
            .enumerate_physical_devices()?
            .filter(|pd| pd.supported_features().contains(&required))
            .filter(|pd| find_compute_family(pd).is_some())
            .collect();
        candidates.sort_by_key(|pd| match pd.properties().device_type {
            PhysicalDeviceType::IntegratedGpu => 0, // the target's 8060S
            PhysicalDeviceType::DiscreteGpu => 1,
            _ => 2,
        });
        let physical = candidates
            .into_iter()
            .next()
            .ok_or(GpuError::NoDevice("f16/16-bit-storage compute GPU"))?;

        let queue_family_index = find_compute_family(&physical).expect("filtered above");

        // Cooperative matrix needs the extension, its feature bit, and the
        // Vulkan memory model (SPIR-V requirement). All present on target;
        // optional so dev machines without them still run non-coopmat tests.
        // (No kernel currently pins a subgroup size; a variant that sets
        // `subgroup_size` in build.rs needs `subgroup_size_control` enabled
        // here.)
        let supports_coopmat = physical.supported_extensions().khr_cooperative_matrix
            && physical.supported_features().cooperative_matrix
            && physical.supported_features().vulkan_memory_model;
        let mut features = required;
        let mut extensions = DeviceExtensions::empty();
        if supports_coopmat {
            extensions.khr_cooperative_matrix = true;
            features.cooperative_matrix = true;
            features.vulkan_memory_model = true;
        }

        let subgroup_size = physical.properties().subgroup_size.unwrap_or(64);
        let (device, mut queues) = Device::new(
            physical,
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_features: features,
                enabled_extensions: extensions,
                ..Default::default()
            },
        )
        .map_err(GpuError::validated)?;
        let queue = queues.next().expect("one queue requested");

        Ok(Self {
            memory_allocator: Arc::new(StandardMemoryAllocator::new_default(device.clone())),
            descriptor_set_allocator: Arc::new(StandardDescriptorSetAllocator::new(
                device.clone(),
                Default::default(),
            )),
            command_buffer_allocator: Arc::new(StandardCommandBufferAllocator::new(
                device.clone(),
                Default::default(),
            )),
            device,
            queue,
            subgroup_size,
            cooperative_matrix: supports_coopmat,
            debug_utils,
        })
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    pub fn memory_allocator(&self) -> &Arc<StandardMemoryAllocator> {
        &self.memory_allocator
    }

    pub fn descriptor_set_allocator(&self) -> &Arc<StandardDescriptorSetAllocator> {
        &self.descriptor_set_allocator
    }

    pub fn command_buffer_allocator(&self) -> &Arc<StandardCommandBufferAllocator> {
        &self.command_buffer_allocator
    }

    pub fn device_name(&self) -> String {
        self.device
            .physical_device()
            .properties()
            .device_name
            .clone()
    }
}

fn find_compute_family(pd: &PhysicalDevice) -> Option<u32> {
    pd.queue_family_properties()
        .iter()
        .position(|q| q.queue_flags.intersects(QueueFlags::COMPUTE))
        .map(|i| i as u32)
}
