//! Vulkan capability enumeration: per-device features, limits we depend on,
//! cooperative-matrix configurations, and memory heaps.

use anyhow::Context;
use serde::Serialize;
use std::sync::Arc;
use vulkano::device::physical::PhysicalDevice;
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::{VulkanLibrary, VulkanObject};

#[derive(Serialize)]
pub struct VulkanReport {
    pub devices: Vec<DeviceReport>,
}

#[derive(Serialize)]
pub struct DeviceReport {
    pub name: String,
    pub device_type: String,
    pub api_version: String,
    pub driver: String,
    pub subgroup_size: Option<u32>,
    pub subgroup_operations: Option<String>,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_compute_work_group_invocations: u32,
    pub features: FeatureReport,
    pub khr_cooperative_matrix: bool,
    /// Supported coopmat M/N/K/type configurations (Debug-formatted; the set
    /// drives which GEMM kernel variants get compiled).
    pub cooperative_matrix_configs: Vec<String>,
    pub memory_heaps: Vec<HeapReport>,
}

/// The features the plans assert at startup (docs/plans/02): all of these
/// must be true on the target device.
#[derive(Serialize)]
pub struct FeatureReport {
    pub shader_float16: bool,
    pub storage_buffer16_bit_access: bool,
    pub uniform_and_storage_buffer16_bit_access: bool,
    pub timeline_semaphore: bool,
    pub cooperative_matrix: bool,
    pub shader_int8: bool,
}

#[derive(Serialize)]
pub struct HeapReport {
    pub size_bytes: u64,
    pub flags: String,
}

pub fn probe() -> anyhow::Result<VulkanReport> {
    let library = VulkanLibrary::new().context("loading Vulkan library")?;
    let instance = Instance::new(library, InstanceCreateInfo::default())
        .context("creating Vulkan instance")?;

    let mut devices = Vec::new();
    for pd in instance
        .enumerate_physical_devices()
        .context("enumerating physical devices")?
    {
        let p = pd.properties();
        let f = pd.supported_features();
        let e = pd.supported_extensions();

        let cooperative_matrix_configs = if e.khr_cooperative_matrix {
            coopmat_configs(&instance, &pd)
        } else {
            Vec::new()
        };

        let memory_heaps = pd
            .memory_properties()
            .memory_heaps
            .iter()
            .map(|h| HeapReport {
                size_bytes: h.size,
                flags: format!("{:?}", h.flags),
            })
            .collect();

        devices.push(DeviceReport {
            name: p.device_name.clone(),
            device_type: format!("{:?}", p.device_type),
            api_version: format!(
                "{}.{}.{}",
                p.api_version.major, p.api_version.minor, p.api_version.patch
            ),
            driver: format!(
                "{} ({})",
                p.driver_name.clone().unwrap_or_default(),
                p.driver_info.clone().unwrap_or_default()
            ),
            subgroup_size: p.subgroup_size,
            subgroup_operations: p
                .subgroup_supported_operations
                .map(|ops| format!("{ops:?}")),
            max_storage_buffer_range: p.max_storage_buffer_range,
            max_push_constants_size: p.max_push_constants_size,
            max_compute_work_group_invocations: p.max_compute_work_group_invocations,
            features: FeatureReport {
                shader_float16: f.shader_float16,
                storage_buffer16_bit_access: f.storage_buffer16_bit_access,
                uniform_and_storage_buffer16_bit_access: f.uniform_and_storage_buffer16_bit_access,
                timeline_semaphore: f.timeline_semaphore,
                cooperative_matrix: f.cooperative_matrix,
                shader_int8: f.shader_int8,
            },
            khr_cooperative_matrix: e.khr_cooperative_matrix,
            cooperative_matrix_configs,
            memory_heaps,
        });
    }

    Ok(VulkanReport { devices })
}

/// Enumerate `VK_KHR_cooperative_matrix` M/N/K/type configurations via the
/// raw Vulkan call (vulkano 0.35 validates coopmat shaders but does not wrap
/// `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`).
fn coopmat_configs(instance: &Arc<Instance>, pd: &Arc<PhysicalDevice>) -> Vec<String> {
    const FN_NAME: &std::ffi::CStr = c"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR";
    type GetProps = unsafe extern "system" fn(
        ash::vk::PhysicalDevice,
        *mut u32,
        *mut ash::vk::CooperativeMatrixPropertiesKHR<'_>,
    ) -> ash::vk::Result;

    // SAFETY: the instance handle is valid and the name is a valid C string.
    let fp = unsafe {
        instance
            .library()
            .get_instance_proc_addr(instance.handle(), FN_NAME.as_ptr())
    };
    let Some(fp) = fp else {
        return vec!["vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR not found".to_owned()];
    };
    // SAFETY: the loader returned this pointer for exactly this PFN name.
    let get_props: GetProps = unsafe { std::mem::transmute(fp) };

    let mut count = 0u32;
    // SAFETY: valid physical-device handle; null properties pointer is the
    // standard count-query form.
    let res = unsafe { get_props(pd.handle(), &mut count, std::ptr::null_mut()) };
    if res != ash::vk::Result::SUCCESS {
        return vec![format!("count query failed: {res:?}")];
    }
    let mut props = vec![ash::vk::CooperativeMatrixPropertiesKHR::default(); count as usize];
    // SAFETY: props has capacity for `count` elements, each properly defaulted
    // (sType set, pNext null).
    let res = unsafe { get_props(pd.handle(), &mut count, props.as_mut_ptr()) };
    if res != ash::vk::Result::SUCCESS {
        return vec![format!("properties query failed: {res:?}")];
    }

    props
        .iter()
        .take(count as usize)
        .map(|p| {
            format!(
                "M{}xN{}xK{} A={:?} B={:?} C={:?} Result={:?} saturating={} scope={:?}",
                p.m_size,
                p.n_size,
                p.k_size,
                p.a_type,
                p.b_type,
                p.c_type,
                p.result_type,
                p.saturating_accumulation != 0,
                p.scope,
            )
        })
        .collect()
}
