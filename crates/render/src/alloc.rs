//! Memory allocator wrapper around `gpu-allocator`'s Vulkan backend.

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;

/// Shared GPU memory allocator. Wrapped in a `Mutex` because `gpu-allocator`
/// requires `&mut self` for allocate/free and we allocate from multiple paths.
pub struct Alloc {
    inner: Mutex<Allocator>,
}

impl Alloc {
    pub fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Result<Self> {
        // Buffer device address is opt-in via VkPhysicalDeviceBufferDeviceAddressFeatures
        // (a Vulkan 1.2 struct). We currently don't use buffer_device_address, so the
        // default of `false` is correct. If a future feature needs it, enable here
        // and add the feature to LogicalDeviceCreateInfo's pNext chain.
        let buffer_device_address = false;
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address,
            allocation_sizes: Default::default(),
        })?;
        Ok(Self {
            inner: Mutex::new(allocator),
        })
    }

    /// Allocate a GPU memory block. See `gpu_allocator::vulkan::AllocationCreateDesc`.
    pub fn allocate(
        &self,
        desc: &AllocationCreateDesc,
    ) -> Result<Allocation> {
        self.inner
            .lock()
            .allocate(desc)
            .map_err(|e| anyhow::anyhow!("allocator allocate failed: {e}"))
    }

    /// Free a previously allocated memory block.
    pub fn free(&self, alloc: Allocation) {
        let _ = self.inner.lock().free(alloc);
    }
}
