//! Device and queue setup on raw `ash`: one instance, one device, one compute
//! queue, with the features the plans assert (plan 02 — all confirmed present
//! on the target box by the M0 probe).
//!
//! [`DeviceCtx`] owns the Vulkan objects whose lifetime everything else hangs
//! off (instance, device, queue). It is `Arc`-shared into every [`Buffer`],
//! [`Kernel`], [`CommandGraph`] and [`GpuTimer`] so the device outlives them and
//! their `Drop`s can destroy their own handles; `DeviceCtx::drop` tears the
//! device/instance down last. This is the lifetime discipline vulkano gave us
//! for free.

use std::ffi::{CStr, c_char};
use std::sync::Arc;

use ash::vk;

use crate::GpuError;

/// The Vulkan objects shared (via `Arc`) by every GPU resource in this crate.
/// Dropping the last `Arc` waits for the device to idle and destroys the
/// device then the instance.
pub(crate) struct DeviceCtx {
    /// Loaded Vulkan library; kept alive for the instance/device.
    _entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: ash::Device,
    pub physical: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub mem_props: vk::PhysicalDeviceMemoryProperties,
    /// Debug-utils device loader for per-dispatch label regions (RGP/SQTT
    /// markers); `None` when `VK_EXT_debug_utils` was unavailable.
    pub debug_utils: Option<ash::ext::debug_utils::Device>,
    pub timestamp_period: f32,
    pub subgroup_size: u32,
    pub cooperative_matrix: bool,
}

impl DeviceCtx {
    /// Pick a memory type satisfying `required` from a buffer's `type_bits`,
    /// preferring `HOST_COHERENT` (no manual flush/invalidate). Returns the
    /// type index and whether it is coherent.
    pub fn find_mem_type(
        &self,
        type_bits: u32,
        required: vk::MemoryPropertyFlags,
    ) -> Option<(u32, bool)> {
        let types = &self.mem_props.memory_types[..self.mem_props.memory_type_count as usize];
        // Prefer a coherent match, then any match.
        for prefer_coherent in [true, false] {
            for (i, mt) in types.iter().enumerate() {
                if type_bits & (1 << i) == 0 {
                    continue;
                }
                if !mt.property_flags.contains(required) {
                    continue;
                }
                let coherent = mt
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                if coherent == prefer_coherent {
                    return Some((i as u32, coherent));
                }
            }
        }
        None
    }
}

impl Drop for DeviceCtx {
    fn drop(&mut self) {
        // SAFETY: all per-resource handles (buffers, pipelines, graphs, timers)
        // hold an Arc<DeviceCtx>, so this runs only after every one of them has
        // been dropped and destroyed its own objects. Wait for the GPU to go
        // idle before tearing the device down.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The Vulkan compute context: one device, one compute queue (the dedicated
/// GPU thread in plan 00 owns this; nothing else submits).
pub struct GpuContext {
    pub(crate) ctx: Arc<DeviceCtx>,
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
        // SAFETY: loads the system Vulkan loader; no outstanding handles yet.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| GpuError::Library(format!("loading Vulkan: {e}")))?;

        // `ext_debug_utils` lets us name each dispatch with a command-buffer
        // label region; RADV emits those as SQTT markers so RGP captures show
        // which kernel each event is. Optional — absence just means unlabelled
        // traces, never a failure.
        let want_debug_utils = instance_has_extension(&entry, ash::ext::debug_utils::NAME)?;

        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_2);
        let mut instance_exts: Vec<*const c_char> = Vec::new();
        if want_debug_utils {
            instance_exts.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        let instance_ci = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&instance_exts);
        // SAFETY: `entry` is valid; the create info borrows live locals.
        let instance = unsafe { entry.create_instance(&instance_ci, None) }
            .map_err(|e| GpuError::Vk(format!("create_instance: {e}")))?;

        // SAFETY: valid instance.
        let physicals = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| GpuError::Vk(format!("enumerate_physical_devices: {e}")))?;

        // Each capable candidate, tagged with its type rank (integrated GPU =
        // the target's 8060S sorts first).
        let mut candidates: Vec<(u32, vk::PhysicalDevice, Caps)> = Vec::new();
        for pd in physicals {
            if let Some(caps) = device_caps(&instance, pd) {
                // SAFETY: valid physical device.
                let props = unsafe { instance.get_physical_device_properties(pd) };
                let rank = match props.device_type {
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 0,
                    vk::PhysicalDeviceType::DISCRETE_GPU => 1,
                    _ => 2,
                };
                candidates.push((rank, pd, caps));
            }
        }
        candidates.sort_by_key(|(rank, ..)| *rank);
        let (_, physical, caps) = candidates
            .into_iter()
            .next()
            .ok_or(GpuError::NoDevice("f16/16-bit-storage compute GPU"))?;

        // Build the enabled-feature pNext chain. Cooperative matrix is
        // optional so dev machines without it still run non-coopmat tests.
        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .shader_float16(true)
            .shader_int8(true)
            .storage_buffer8_bit_access(true)
            .timeline_semaphore(true)
            .vulkan_memory_model(true);
        let mut coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default()
            .cooperative_matrix(true);

        let mut device_exts: Vec<*const c_char> = Vec::new();
        if caps.cooperative_matrix {
            device_exts.push(ash::khr::cooperative_matrix::NAME.as_ptr());
        }

        let priorities = [1.0f32];
        let queue_ci = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(caps.compute_family)
            .queue_priorities(&priorities);
        let queue_cis = [queue_ci];

        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut features11)
            .push_next(&mut features12);
        if caps.cooperative_matrix {
            features2 = features2.push_next(&mut coopmat);
        }
        let device_ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_cis)
            .enabled_extension_names(&device_exts)
            .push_next(&mut features2);

        // SAFETY: valid instance + physical device; create info borrows live
        // locals (the feature chain is not moved before this call returns).
        let device = unsafe { instance.create_device(physical, &device_ci, None) }
            .map_err(|e| GpuError::Vk(format!("create_device: {e}")))?;

        // SAFETY: the family/index were just created above.
        let queue = unsafe { device.get_device_queue(caps.compute_family, 0) };
        // SAFETY: valid physical device.
        let mem_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let debug_utils =
            want_debug_utils.then(|| ash::ext::debug_utils::Device::new(&instance, &device));

        let ctx = Arc::new(DeviceCtx {
            _entry: entry,
            instance,
            device,
            physical,
            queue,
            queue_family: caps.compute_family,
            mem_props,
            debug_utils,
            timestamp_period: caps.timestamp_period,
            subgroup_size: caps.subgroup_size,
            cooperative_matrix: caps.cooperative_matrix,
        });

        Ok(Self {
            subgroup_size: ctx.subgroup_size,
            cooperative_matrix: ctx.cooperative_matrix,
            debug_utils: ctx.debug_utils.is_some(),
            ctx,
        })
    }

    pub(crate) fn device(&self) -> &ash::Device {
        &self.ctx.device
    }

    pub fn device_name(&self) -> String {
        // SAFETY: valid physical device.
        let props = unsafe {
            self.ctx
                .instance
                .get_physical_device_properties(self.ctx.physical)
        };
        let name = props.device_name_as_c_str().unwrap_or(c"<unknown>");
        name.to_string_lossy().into_owned()
    }
}

/// Capabilities of a candidate device that passed the feature gate.
struct Caps {
    compute_family: u32,
    subgroup_size: u32,
    timestamp_period: f32,
    cooperative_matrix: bool,
}

/// Return `Some(Caps)` if `pd` has a compute queue and every required feature;
/// `None` otherwise (so the device is filtered out). Cooperative matrix is not
/// required — its presence is recorded in `Caps`.
fn device_caps(instance: &ash::Instance, pd: vk::PhysicalDevice) -> Option<Caps> {
    // SAFETY: valid instance + physical device throughout.
    let compute_family = unsafe { instance.get_physical_device_queue_family_properties(pd) }
        .iter()
        .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))? as u32;

    let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut coopmat = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut features11)
        .push_next(&mut features12)
        .push_next(&mut coopmat);
    // SAFETY: valid physical device; the chain outlives the call.
    unsafe { instance.get_physical_device_features2(pd, &mut features2) };

    let required = features11.storage_buffer16_bit_access == vk::TRUE
        && features11.uniform_and_storage_buffer16_bit_access == vk::TRUE
        && features12.shader_float16 == vk::TRUE
        && features12.shader_int8 == vk::TRUE
        && features12.storage_buffer8_bit_access == vk::TRUE
        && features12.timeline_semaphore == vk::TRUE;
    if !required {
        return None;
    }

    // Cooperative matrix needs the extension, its feature bit, and the Vulkan
    // memory model (a SPIR-V requirement for the coopmat ops).
    let has_coopmat_ext = device_has_extension(instance, pd, ash::khr::cooperative_matrix::NAME);
    let cooperative_matrix = has_coopmat_ext
        && coopmat.cooperative_matrix == vk::TRUE
        && features12.vulkan_memory_model == vk::TRUE;

    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    // SAFETY: valid physical device; chain outlives the call.
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };
    // Read out of `props2` before touching `subgroup` (props2 borrows it mutably).
    let timestamp_period = props2.properties.limits.timestamp_period;
    let subgroup_size = if subgroup.subgroup_size > 0 {
        subgroup.subgroup_size
    } else {
        64
    };

    Some(Caps {
        compute_family,
        subgroup_size,
        timestamp_period,
        cooperative_matrix,
    })
}

fn instance_has_extension(entry: &ash::Entry, name: &CStr) -> Result<bool, GpuError> {
    // SAFETY: valid entry; `None` layer queries the implementation extensions.
    let props = unsafe { entry.enumerate_instance_extension_properties(None) }
        .map_err(|e| GpuError::Vk(format!("enumerate_instance_extension_properties: {e}")))?;
    Ok(props
        .iter()
        .any(|p| p.extension_name_as_c_str() == Ok(name)))
}

fn device_has_extension(instance: &ash::Instance, pd: vk::PhysicalDevice, name: &CStr) -> bool {
    // SAFETY: valid instance + physical device.
    let Ok(props) = (unsafe { instance.enumerate_device_extension_properties(pd) }) else {
        return false;
    };
    props
        .iter()
        .any(|p| p.extension_name_as_c_str() == Ok(name))
}
