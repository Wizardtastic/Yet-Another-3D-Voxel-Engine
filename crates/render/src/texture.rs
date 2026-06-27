//! Atlas texture: uploads the procedurally generated RGBA8 atlas to a
//! device-local image and creates a linear sampler for it.

use anyhow::{anyhow, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::alloc::Alloc;
use crate::atlas::Atlas;
use crate::buffer::{create_image_view, GpuBuffer};

/// Owned atlas image + sampler. Destroy via `destroy`.
pub struct AtlasTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    allocation: Option<gpu_allocator::vulkan::Allocation>,
}

impl AtlasTexture {
    pub fn destroy(mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }

    /// Destroy in place (for use in `Drop` where the value can't be moved out).
    pub fn destroy_in_place(&mut self, device: &ash::Device, alloc: &Alloc) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            self.sampler = vk::Sampler::null();
            self.view = vk::ImageView::null();
            self.image = vk::Image::null();
        }
        if let Some(a) = self.allocation.take() {
            alloc.free(a);
        }
    }

    /// Create and upload the atlas. `pool` and `queue` are used for a one-time
    /// staging copy + layout transition.
    pub fn new(
        device: &ash::Device,
        alloc: &Alloc,
        pool: vk::CommandPool,
        queue: vk::Queue,
        atlas: &Atlas,
    ) -> Result<Self> {
        let format = vk::Format::R8G8B8A8_UNORM;
        let extent = vk::Extent3D {
            width: atlas.width,
            height: atlas.height,
            depth: 1,
        };

        let create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let image = unsafe { device.create_image(&create_info, None) }
            .map_err(|e| anyhow!("atlas create_image failed: {e:?}"))?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = alloc
            .allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
                name: "atlas",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
            })?;
        unsafe {
            device
                .bind_image_memory(image, allocation.memory(), allocation.offset())
                .map_err(|e| anyhow!("atlas bind failed: {e:?}"))?;
        }

        // Staging buffer + copy.
        let bytes = atlas.rgba.len() as vk::DeviceSize;
        let mut staging = GpuBuffer::host_visible(
            device,
            alloc,
            bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            "atlas_staging",
        )?;
        staging.upload(&atlas.rgba)?;

        // One-time command buffer for: transition UNDEFINED->TRANSFER_DST, copy,
        // then TRANSFER_DST->SHADER_READ_ONLY.
        let cmd = begin_one_time(device, pool)?;
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(atlas.width)
            .buffer_image_height(atlas.height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(extent);
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );
        end_and_submit(device, pool, queue, cmd)?;

        // Staging buffer can be freed after the submit completes (we wait on the queue).
        staging.destroy(device, alloc);

        let view = create_image_view(device, image, format, vk::ImageAspectFlags::COLOR)?;

        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::INT_TRANSPARENT_BLACK)
            .unnormalized_coordinates(false);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }
            .map_err(|e| anyhow!("create_sampler failed: {e:?}"))?;

        Ok(Self {
            image,
            view,
            sampler,
            allocation: Some(allocation),
        })
    }
}

/// Allocate + begin a one-time-submit command buffer.
pub fn begin_one_time(device: &ash::Device, pool: vk::CommandPool) -> Result<vk::CommandBuffer> {
    let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
        .map_err(|e| anyhow!("allocate_command_buffers failed: {e:?}"))?[0];
    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(cmd, &begin_info)
            .map_err(|e| anyhow!("begin_command_buffer failed: {e:?}"))?;
    }
    Ok(cmd)
}

/// End, submit, and wait on a one-time command buffer, then free it.
pub fn end_and_submit(
    device: &ash::Device,
    pool: vk::CommandPool,
    queue: vk::Queue,
    cmd: vk::CommandBuffer,
) -> Result<()> {
    unsafe {
        device
            .end_command_buffer(cmd)
            .map_err(|e| anyhow!("end_command_buffer failed: {e:?}"))?;
        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        let submit_infos = [submit_info];
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(|e| anyhow!("create_fence failed: {e:?}"))?;
        device
            .queue_submit(queue, &submit_infos, fence)
            .map_err(|e| anyhow!("queue_submit failed: {e:?}"))?;
        device
            .wait_for_fences(&[fence], true, u64::MAX)
            .map_err(|e| anyhow!("wait_for_fences failed: {e:?}"))?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, &[cmd]);
    }
    Ok(())
}

/// Record an image memory barrier transitioning `image` between layouts.
#[allow(clippy::too_many_arguments)]
pub fn transition_image_layout(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    aspect: vk::ImageAspectFlags,
    mip_levels: u32,
    layer_count: u32,
) {
    let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        ),
        (vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::PRESENT_SRC_KHR, vk::ImageLayout::TRANSFER_SRC_OPTIMAL) => (
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::PRESENT_SRC_KHR) => (
            vk::AccessFlags::TRANSFER_READ,
            vk::AccessFlags::MEMORY_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        ),
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        ),
        (
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ) => (
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        (
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        ) => (
            vk::AccessFlags::SHADER_READ,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        ),
        (
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ) => (
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => {
            log::warn!(
                "unhandled image layout transition: {:?} -> {:?}",
                old_layout,
                new_layout
            );
            (
                vk::AccessFlags::empty(),
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            )
        }
    };

    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: aspect,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count,
        })
        .src_access_mask(src_access)
        .dst_access_mask(dst_access);
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }
}
