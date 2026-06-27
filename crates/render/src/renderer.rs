//! The `Renderer` facade: owns the entire Vulkan session and exposes a tiny
//! API the engine drives (`new`, `upload_chunks`, `remove_chunk`, `draw_frame`,
//! `capture_frame`, `resize`, `Drop`).
//!
//! Internal structure (all Vulkan handles live on `Renderer`):
//! - instance + optional debug messenger
//! - surface (from a raw window handle via `ash-window`)
//! - physical device + queue families (graphics + present, possibly one)
//! - logical device + graphics/present queues
//! - memory allocator (`gpu-allocator`)
//! - swapchain + image views + depth image + framebuffers
//! - render pass + chunk graphics pipeline + pipeline layout + descriptor sets
//! - atlas texture + sampler
//! - per-frame (×2 in flight): command buffer, fences/semaphores, camera UBO
//! - per-chunk: vertex + index buffers in a `HashMap`

use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use ash::vk;
use ash::{Entry, Instance as AshInstance};
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use parking_lot::RwLock;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use std::path::Path;

use voxel_core::{
    math::{chunk_origin, ChunkPos},
    Camera, Frustum,
};

use crate::alloc::Alloc;
use crate::atlas::build_atlas_with_textures;
use crate::buffer::{create_image_view, GpuBuffer, GpuImage};
use crate::texture::{begin_one_time, end_and_submit, transition_image_layout, AtlasTexture};
use crate::ui::UiDrawData;

const FRAMES_IN_FLIGHT: usize = 2;
const GPU_TIMESTAMP_COUNT: u32 = 8; // frame_start, shadow_end, sky_end, opaque_end, transparent_end, ui_end, main_pass_end, post_end

/// GPU timing results for a single frame (in milliseconds).
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimings {
    pub frame_ms: f32,
    pub shadow_ms: f32,
    pub sky_ms: f32,
    pub opaque_ms: f32,
    pub transparent_ms: f32,
    pub ui_ms: f32,
    pub post_ms: f32,
}

/// Which render pass a chunk mesh belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MeshPass {
    Opaque,
    Transparent,
}

/// Vertex + index data for one chunk pass, as raw bytes ready to upload.
pub struct ChunkUpload {
    pub pos: ChunkPos,
    pub pass: MeshPass,
    /// Vertex bytes (24 bytes each, layout = [`crate::Vertex`]).
    pub vertices: Vec<u8>,
    /// Index bytes (4 bytes each, `u32`).
    pub indices: Vec<u8>,
    pub index_count: u32,
}

/// Renderer configuration.
#[derive(Clone, Debug)]
pub struct RendererConfig {
    /// Enable Vulkan validation layers + debug messenger (debug builds).
    pub validation: bool,
    /// Clear colour (sky) in linear RGB 0..1.
    pub clear_color: [f32; 4],
    /// Use FIFO (vsync) present mode. If false, prefer MAILBOX.
    pub vsync: bool,
    /// Fog colour (linear RGB) and density (unused placeholder).
    pub fog_color: [f32; 3],
    /// Distance at which fog fully obscures chunks.
    pub fog_distance: f32,
    /// Directory containing PNG texture overrides (filenames `<tile_index>.png`).
    /// If `None` or the directory doesn't exist, the procedural atlas is used.
    pub textures_dir: Option<std::path::PathBuf>,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            validation: false,
            clear_color: [0.52, 0.72, 0.95, 1.0],
            vsync: true,
            fog_color: [0.62, 0.80, 0.96],
            fog_distance: 320.0,
            textures_dir: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct QueueFamilies {
    graphics: u32,
    present: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct CameraUbo {
    /// xyz = camera position, w = fog max distance.
    cam_pos_and_maxdist: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct FogUbo {
    /// rgb = fog colour, a = unused.
    color_and_density: [f32; 4],
    /// x = ambient brightness (day/night), yzw = sun direction (for future use).
    ambient_and_sun: [f32; 4],
}

/// Per-frame sky uniform: horizon colour, zenith colour, sun direction.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SkyUbo {
    horizon: [f32; 4],
    zenith: [f32; 4],
    sun_dir: [f32; 4],
}

/// Per-frame shadow uniform for cascaded shadow maps (binding 4 of the chunk set).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct ShadowUbo {
    cascade_vps: [[f32; 16]; 4],
    cascade_splits: [f32; 4],
    light_dir_and_bias: [f32; 4],
}

/// Per-chunk GPU buffers for a single render pass.
struct PassBuffers {
    vbo: GpuBuffer,
    ibo: GpuBuffer,
    index_count: u32,
}

/// Per-chunk GPU buffers, split by render pass. A chunk at a water-land border
/// has both; a purely inland chunk has only opaque.
struct ChunkBuffers {
    opaque: Option<PassBuffers>,
    transparent: Option<PassBuffers>,
}

impl ChunkBuffers {
    fn new() -> Self {
        Self {
            opaque: None,
            transparent: None,
        }
    }

    fn destroy(self, device: &ash::Device, alloc: &Alloc) {
        if let Some(b) = self.opaque {
            b.vbo.destroy(device, alloc);
            b.ibo.destroy(device, alloc);
        }
        if let Some(b) = self.transparent {
            b.vbo.destroy(device, alloc);
            b.ibo.destroy(device, alloc);
        }
    }
}

struct Frame {
    cmd: vk::CommandBuffer,
    in_flight_fence: vk::Fence,
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    camera_ubo: GpuBuffer,
    shadow_ubo: GpuBuffer,
    descriptor_set: vk::DescriptorSet,
}

pub struct Renderer {
    config: RendererConfig,
    _entry: Entry,
    instance: AshInstance,
    #[allow(dead_code)]
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    #[allow(dead_code)]
    queues: QueueFamilies,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    surface: vk::SurfaceKHR,
    surface_instance: ash::khr::surface::Instance,
    swapchain_device: ash::khr::swapchain::Device,
    alloc: ManuallyDrop<Arc<Alloc>>,

    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    depth: Option<GpuImage>,
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    wireframe_pipeline: vk::Pipeline,
    transparent_pipeline: vk::Pipeline,
    wireframe_enabled: bool,
    #[allow(dead_code)]
    descriptor_pool: vk::DescriptorPool,
    #[allow(dead_code)]
    descriptor_set_layout: vk::DescriptorSetLayout,

    command_pool: vk::CommandPool,
    #[allow(dead_code)]
    atlas: AtlasTexture,
    #[allow(dead_code)]
    fog_ubo: GpuBuffer,

    // ── UI pipeline ──
    ui_pipeline: vk::Pipeline,
    ui_pipeline_layout: vk::PipelineLayout,
    #[allow(dead_code)]
    ui_descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    ui_descriptor_pool: vk::DescriptorPool,
    ui_descriptor_set: vk::DescriptorSet,
    #[allow(dead_code)]
    font_texture: AtlasTexture,
    ui_vbo: GpuBuffer,
    ui_ibo: GpuBuffer,

    // ── Sky pipeline ──
    sky_pipeline: vk::Pipeline,
    sky_pipeline_layout: vk::PipelineLayout,
    #[allow(dead_code)]
    sky_descriptor_set_layout: vk::DescriptorSetLayout,
    #[allow(dead_code)]
    sky_descriptor_pool: vk::DescriptorPool,
    sky_descriptor_set: vk::DescriptorSet,
    sky_ubo: GpuBuffer,

    // ── Shadow pass ──
    shadow_render_pass: vk::RenderPass,
    shadow_pipeline: vk::Pipeline,
    shadow_pipeline_layout: vk::PipelineLayout,
    shadow_image: GpuImage,
    shadow_layer_views: Vec<vk::ImageView>,
    shadow_sampler: vk::Sampler,
    shadow_framebuffers: Vec<vk::Framebuffer>,
    shadow_ubo_data: ShadowUbo,

    // ── Offscreen color (for post-processing) ──
    offscreen_images: Vec<GpuImage>,
    offscreen_framebuffers: Vec<vk::Framebuffer>,

    // ── Post pass ──
    post_render_pass: vk::RenderPass,
    post_pipeline: vk::Pipeline,
    post_pipeline_layout: vk::PipelineLayout,
    post_descriptor_set_layout: vk::DescriptorSetLayout,
    post_descriptor_pool: vk::DescriptorPool,
    post_descriptor_sets: Vec<vk::DescriptorSet>,
    post_sampler: vk::Sampler,
    post_framebuffers: Vec<vk::Framebuffer>,
    post_params: [f32; 4],

    frames: Vec<Frame>,
    chunks: RwLock<HashMap<ChunkPos, ChunkBuffers>>,

    // ── GPU timers ──
    query_pool: vk::QueryPool,
    timestamp_period: f32,
    timings: GpuTimings,

    /// Set when the window was resized; swapchain is recreated next draw.
    needs_resize: bool,
    frame_counter: usize,
    /// Dynamic sky params set by the engine each frame for day/night.
    sky_horizon: [f32; 3],
    sky_zenith: [f32; 3],
    sky_fog: [f32; 3],
    sky_ambient: f32,
    sky_underwater: bool,
    sun_dir: [f32; 3],
}

impl Renderer {
    /// Create a complete renderer for `window`.
    pub fn new(
        window_handle: RawWindowHandle,
        display_handle: RawDisplayHandle,
        config: RendererConfig,
    ) -> Result<Self> {
        let entry = unsafe { Entry::load() }.map_err(|e| anyhow!("Vulkan loader: {e}"))?;

        // --- instance ---
        let instance = create_instance(&entry, display_handle, config.validation)?;

        let debug_messenger = if config.validation {
            create_debug_messenger(&entry, &instance).ok()
        } else {
            None
        };

        // --- surface ---
        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
        }
        .map_err(|e| anyhow!("create_surface: {e:?}"))?;
        let surface_instance = ash::khr::surface::Instance::new(&entry, &instance);

        // --- physical device + queues ---
        let (physical_device, queues) =
            pick_physical_device(&instance, &surface_instance, surface)?;

        // --- logical device ---
        let (device, graphics_queue, present_queue) = create_logical_device(
            &instance,
            physical_device,
            queues,
            &surface_instance,
            surface,
        )?;
        let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);

        let alloc = ManuallyDrop::new(Arc::new(Alloc::new(&instance, physical_device, &device)?));

        // --- command pool ---
        let pool_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queues.graphics);
        let command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .context("create_command_pool")?;

        // --- atlas texture ---
        // The textures directory must exist and contain a textures.toml
        // config. If it doesn't, all tiles show the blue+black error pattern.
        let atlas_pixels = match config.textures_dir.as_deref() {
            Some(dir) if dir.is_dir() => build_atlas_with_textures(dir),
            _ => {
                log::warn!(
                    "no textures_dir configured or directory not found — all tiles will show error pattern"
                );
                build_atlas_with_textures(Path::new(""))
            }
        };
        let atlas =
            AtlasTexture::new(&device, &alloc, command_pool, graphics_queue, &atlas_pixels)?;

        // --- descriptor set layout + pool + fog UBO ---
        let descriptor_set_layout = create_descriptor_set_layout(&device)?;
        let descriptor_pool = create_descriptor_pool(&device, FRAMES_IN_FLIGHT)?;
        let mut fog_ubo = GpuBuffer::host_visible(
            &device,
            &alloc,
            std::mem::size_of::<FogUbo>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "fog_ubo",
        )?;
        {
            let fog = FogUbo {
                color_and_density: [
                    config.fog_color[0],
                    config.fog_color[1],
                    config.fog_color[2],
                    1.0,
                ],
                ambient_and_sun: [1.0, 0.0, 1.0, 0.0], // full daylight, sun straight up
            };
            let slice = fog_ubo.mapped_slice_mut()?;
            let bytes: &[u8] = bytemuck::bytes_of(&fog);
            slice[..bytes.len()].copy_from_slice(bytes);
        }

        // --- render pass ---
        // We need the swapchain before framebuffers, and the render pass before
        // the pipeline. Build swapchain first.
        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) = create_swapchain(
            &device,
            &swapchain_device,
            &surface_instance,
            physical_device,
            surface,
            config.vsync,
        )?;
        let swapchain_image_views =
            create_image_views(&device, &swapchain_images, swapchain_format)?;

        let depth_format = find_depth_format(&instance, physical_device);
        let render_pass = create_render_pass(&device, swapchain_format, depth_format)?;
        let pipeline_layout = create_pipeline_layout(&device, descriptor_set_layout)?;
        let pipeline = create_graphics_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            vk::PolygonMode::FILL,
            vk::CullModeFlags::BACK,
        )?;
        let wireframe_pipeline = create_graphics_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            vk::PolygonMode::LINE,
            vk::CullModeFlags::BACK,
        )?;
        let transparent_pipeline = create_graphics_pipeline(
            &device,
            render_pass,
            pipeline_layout,
            vk::PolygonMode::FILL,
            vk::CullModeFlags::NONE,
        )?;

        let depth = GpuImage::depth(&device, &alloc, swapchain_extent, depth_format)?;

        // ── Offscreen color images (post-processing source) ──
        // The main render pass writes to these instead of the swapchain images.
        let mut offscreen_images = Vec::with_capacity(swapchain_images.len());
        for i in 0..swapchain_images.len() {
            let img = GpuImage::color_attachment(
                &device,
                &alloc,
                swapchain_extent,
                swapchain_format,
                "offscreen",
            )?;
            let cmd_init = begin_one_time(&device, command_pool)?;
            transition_image_layout(
                &device,
                cmd_init,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            end_and_submit(&device, command_pool, graphics_queue, cmd_init)?;
            offscreen_images.push(img);
            let _ = i;
        }
        let offscreen_framebuffers = offscreen_images
            .iter()
            .map(|img| {
                create_framebuffer_with(
                    &device,
                    render_pass,
                    &[img.view, depth.view],
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // ── Shadow pass init ──
        let shadow_extent = vk::Extent2D {
            width: 2048,
            height: 2048,
        };
        let shadow_render_pass = create_shadow_render_pass(&device, depth_format)?;
        let shadow_pipeline_layout = create_shadow_pipeline_layout(&device)?;
        let shadow_pipeline =
            create_shadow_pipeline(&device, shadow_render_pass, shadow_pipeline_layout)?;
        let shadow_image = GpuImage::depth_array(
            &device,
            &alloc,
            shadow_extent,
            depth_format,
            4,
            "shadow_map",
        )?;
        let shadow_layer_views: Vec<vk::ImageView> = (0..4u32)
            .map(|i| create_shadow_layer_view(&device, shadow_image.image, depth_format, i))
            .collect::<Result<Vec<_>>>()?;
        let shadow_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .compare_enable(true)
                    .compare_op(vk::CompareOp::LESS_OR_EQUAL)
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_shadow_sampler: {e:?}"))?;
        let shadow_framebuffers: Vec<vk::Framebuffer> = (0..4u32)
            .map(|i| {
                create_framebuffer_with(
                    &device,
                    shadow_render_pass,
                    &[shadow_layer_views[i as usize]],
                    shadow_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        {
            let cmd_init = begin_one_time(&device, command_pool)?;
            transition_image_layout(
                &device,
                cmd_init,
                shadow_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                1,
                4,
            );
            end_and_submit(&device, command_pool, graphics_queue, cmd_init)?;
        }

        // --- per-frame resources ---
        let descriptor_sets = allocate_descriptor_sets(
            &device,
            descriptor_pool,
            descriptor_set_layout,
            FRAMES_IN_FLIGHT,
        )?;
        let mut frames = Vec::with_capacity(FRAMES_IN_FLIGHT);
        for &descriptor_set in descriptor_sets.iter().take(FRAMES_IN_FLIGHT) {
            let cmd = {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                unsafe { device.allocate_command_buffers(&alloc_info) }?[0]
            };
            let in_flight_fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }?;
            let image_available =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            let render_finished =
                unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
            let camera_ubo = GpuBuffer::host_visible(
                &device,
                &alloc,
                std::mem::size_of::<CameraUbo>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "camera_ubo",
            )?;
            let shadow_ubo = GpuBuffer::host_visible(
                &device,
                &alloc,
                std::mem::size_of::<ShadowUbo>() as vk::DeviceSize,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                "shadow_ubo",
            )?;
            // Bind this frame's camera UBO + shared atlas + shared fog + shadow map + shadow UBO.
            update_descriptor_set(
                &device,
                descriptor_set,
                camera_ubo.buffer,
                fog_ubo.buffer,
                atlas.view,
                atlas.sampler,
                shadow_image.view,
                shadow_sampler,
                shadow_ubo.buffer,
            );
            frames.push(Frame {
                cmd,
                in_flight_fence,
                image_available,
                render_finished,
                camera_ubo,
                shadow_ubo,
                descriptor_set,
            });
        }

        // ── UI pipeline ──
        let font = crate::ui::FontAtlas::new();
        let font_texture =
            AtlasTexture::new(&device, &alloc, command_pool, graphics_queue, &font.atlas)?;
        let ui_descriptor_set_layout = create_ui_descriptor_set_layout(&device)?;
        // Separate pool for the UI descriptor set (2 image samplers, 1 set).
        let ui_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 2,
        }];
        let ui_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&ui_pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let ui_descriptor_pool = unsafe { device.create_descriptor_pool(&ui_pool_info, None) }
            .map_err(|e| anyhow!("create_ui_descriptor_pool: {e:?}"))?;
        let ui_descriptor_set =
            allocate_ui_descriptor_set(&device, ui_descriptor_pool, ui_descriptor_set_layout)?;
        update_ui_descriptor_set(
            &device,
            ui_descriptor_set,
            atlas.view,
            atlas.sampler,
            font_texture.view,
            font_texture.sampler,
        );
        let ui_pipeline_layout = create_ui_pipeline_layout(&device, ui_descriptor_set_layout)?;
        let ui_pipeline = create_ui_pipeline(&device, render_pass, ui_pipeline_layout)?;

        // Persistent host-visible buffers for UI vertices/indices (re-uploaded
        // each frame). 256 KB each is way more than a simple HUD needs.
        let ui_vbo = GpuBuffer::host_visible(
            &device,
            &alloc,
            256 * 1024,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            "ui_vbo",
        )?;
        let ui_ibo = GpuBuffer::host_visible(
            &device,
            &alloc,
            256 * 1024,
            vk::BufferUsageFlags::INDEX_BUFFER,
            "ui_ibo",
        )?;

        // ── Sky pipeline ──
        let sky_descriptor_set_layout = create_sky_descriptor_set_layout(&device)?;
        let sky_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];
        let sky_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&sky_pool_sizes)
            .max_sets(1)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let sky_descriptor_pool = unsafe { device.create_descriptor_pool(&sky_pool_info, None) }
            .map_err(|e| anyhow!("create_sky_descriptor_pool: {e:?}"))?;
        let sky_layouts = [sky_descriptor_set_layout];
        let sky_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(sky_descriptor_pool)
            .set_layouts(&sky_layouts);
        let sky_sets = unsafe { device.allocate_descriptor_sets(&sky_alloc_info) }
            .map_err(|e| anyhow!("allocate_sky_descriptor_set: {e:?}"))?;
        let sky_descriptor_set = sky_sets[0];
        let sky_ubo = GpuBuffer::host_visible(
            &device,
            &alloc,
            std::mem::size_of::<SkyUbo>() as vk::DeviceSize,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            "sky_ubo",
        )?;
        // Bind sky UBO to the descriptor set.
        let sky_buf_info = vk::DescriptorBufferInfo::default()
            .buffer(sky_ubo.buffer)
            .offset(0)
            .range(std::mem::size_of::<SkyUbo>() as u64);
        let sky_buf_infos = [sky_buf_info];
        let sky_writes = [vk::WriteDescriptorSet::default()
            .dst_set(sky_descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&sky_buf_infos)];
        unsafe { device.update_descriptor_sets(&sky_writes, &[]) };
        let sky_pipeline_layout = create_sky_pipeline_layout(&device, sky_descriptor_set_layout)?;
        let sky_pipeline = create_sky_pipeline(&device, render_pass, sky_pipeline_layout)?;

        // ── Post pass init ──
        let post_render_pass = create_post_render_pass(&device, swapchain_format)?;
        let post_descriptor_set_layout = create_post_descriptor_set_layout(&device)?;
        let post_pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: swapchain_images.len() as u32,
        }];
        let post_pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&post_pool_sizes)
            .max_sets(swapchain_images.len() as u32)
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
        let post_descriptor_pool = unsafe { device.create_descriptor_pool(&post_pool_info, None) }
            .map_err(|e| anyhow!("create_post_descriptor_pool: {e:?}"))?;
        let post_layouts = vec![post_descriptor_set_layout; swapchain_images.len()];
        let post_alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(post_descriptor_pool)
            .set_layouts(&post_layouts);
        let post_sets = unsafe { device.allocate_descriptor_sets(&post_alloc_info) }
            .map_err(|e| anyhow!("allocate_post_descriptor_sets: {e:?}"))?;
        let post_sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )
        }
        .map_err(|e| anyhow!("create_post_sampler: {e:?}"))?;
        let post_descriptor_sets: Vec<vk::DescriptorSet> = post_sets;
        {
            let img_infos: Vec<vk::DescriptorImageInfo> = post_descriptor_sets
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    vk::DescriptorImageInfo::default()
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image_view(offscreen_images[i].view)
                        .sampler(post_sampler)
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = post_descriptor_sets
                .iter()
                .enumerate()
                .map(|(i, &set)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&img_infos[i..i + 1])
                })
                .collect();
            unsafe { device.update_descriptor_sets(&writes, &[]) };
        }
        let post_pipeline_layout = create_post_pipeline_layout(&device, post_descriptor_set_layout)?;
        let post_pipeline = create_post_pipeline(&device, post_render_pass, post_pipeline_layout)?;
        let post_framebuffers: Vec<vk::Framebuffer> = swapchain_image_views
            .iter()
            .map(|&view| {
                create_framebuffer_with(&device, post_render_pass, &[view], swapchain_extent)
            })
            .collect::<Result<Vec<_>>>()?;

        // ── GPU timestamp query pool ──
        let limits = unsafe { instance.get_physical_device_properties(physical_device) }.limits;
        let timestamp_period = limits.timestamp_period;
        let query_pool_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(GPU_TIMESTAMP_COUNT * FRAMES_IN_FLIGHT as u32);
        let query_pool = unsafe { device.create_query_pool(&query_pool_info, None) }
            .map_err(|e| anyhow!("create_query_pool: {e:?}"))?;

        Ok(Self {
            config,
            _entry: entry,
            instance,
            debug_messenger,
            physical_device,
            device,
            queues,
            graphics_queue,
            present_queue,
            surface,
            surface_instance,
            swapchain_device,
            alloc,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
            depth: Some(depth),
            offscreen_framebuffers,
            render_pass,
            pipeline_layout,
            pipeline,
            wireframe_pipeline,
            transparent_pipeline,
            wireframe_enabled: false,
            descriptor_pool,
            descriptor_set_layout,
            command_pool,
            atlas,
            fog_ubo,
            ui_pipeline,
            ui_pipeline_layout,
            ui_descriptor_set_layout,
            ui_descriptor_pool,
            ui_descriptor_set,
            font_texture,
            ui_vbo,
            ui_ibo,
            sky_pipeline,
            sky_pipeline_layout,
            sky_descriptor_set_layout,
            sky_descriptor_pool,
            sky_descriptor_set,
            sky_ubo,
            shadow_render_pass,
            shadow_pipeline,
            shadow_pipeline_layout,
            shadow_image,
            shadow_layer_views,
            shadow_sampler,
            shadow_framebuffers,
            shadow_ubo_data: ShadowUbo::default(),
            offscreen_images,
            post_render_pass,
            post_pipeline,
            post_pipeline_layout,
            post_descriptor_set_layout,
            post_descriptor_pool,
            post_descriptor_sets,
            post_sampler,
            post_framebuffers,
            post_params: [1.0, 0.0, 0.0, 0.0],
            frames,
            chunks: RwLock::new(HashMap::new()),
            query_pool,
            timestamp_period,
            timings: GpuTimings::default(),
            needs_resize: false,
            frame_counter: 0,
            sky_horizon: [0.52, 0.72, 0.95],
            sky_zenith: [0.35, 0.55, 0.90],
            sky_fog: [0.62, 0.80, 0.96],
            sky_ambient: 1.0,
            sky_underwater: false,
            sun_dir: [0.0, 1.0, 0.0],
        })
    }

    pub fn config(&self) -> &RendererConfig {
        &self.config
    }

    /// Mark the swapchain for recreation on the next draw (call on window resize).
    pub fn resize(&mut self) {
        self.needs_resize = true;
    }

    /// Current swapchain extent (window drawable size in pixels).
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain_extent
    }

    /// Latest GPU timing results (1-2 frame lag).
    pub fn latest_timings(&self) -> GpuTimings {
        self.timings
    }

    /// Set dynamic sky parameters for the day/night cycle. Called each frame
    /// by the engine. Updates the fog UBO's colour + ambient brightness, and
    /// the clear colour used for the sky.
    pub fn set_sky(
        &mut self,
        horizon: [f32; 3],
        zenith: [f32; 3],
        fog: [f32; 3],
        ambient: f32,
        underwater: bool,
    ) {
        self.sky_horizon = horizon;
        self.sky_zenith = zenith;
        self.sky_fog = fog;
        self.sky_ambient = ambient;
        self.sky_underwater = underwater;

        // Update the clear colour to match the horizon sky colour, dimmed by ambient.
        if underwater {
            self.config.clear_color = [0.01, 0.05, 0.15, 1.0];
        } else {
            self.config.clear_color = [
                horizon[0] * ambient.max(0.05),
                horizon[1] * ambient.max(0.05),
                horizon[2] * ambient.max(0.05),
                1.0,
            ];
        }
    }

    /// Set the sun direction explicitly (called by the engine with the day params).
    pub fn set_sun_dir(&mut self, dir: [f32; 3]) {
        self.sun_dir = dir;
    }

    /// Set cascaded shadow map data (4 light-space VP matrices + cascade splits).
    pub fn set_shadow_data(
        &mut self,
        cascade_vps: [[f32; 16]; 4],
        cascade_splits: [f32; 4],
        light_dir_and_bias: [f32; 4],
    ) {
        self.shadow_ubo_data = ShadowUbo {
            cascade_vps,
            cascade_splits,
            light_dir_and_bias,
        };
    }

    /// Set post-processing parameters (exposure, vignette strength, time).
    pub fn set_post_params(&mut self, exposure: f32, vignette: f32, time: f32) {
        self.post_params = [exposure, vignette, time, 0.0];
    }

    /// Flush pending sky/fog/sun UBO data to the GPU.
    /// Must be called after the frame fence wait to avoid data races.
    fn flush_pending_ubos(&mut self) {
        // Fog UBO
        let (fog_color, ambient_val) = if self.sky_underwater {
            ([0.05, 0.15, 0.35], self.sky_ambient * 0.6)
        } else {
            (self.sky_fog, self.sky_ambient)
        };
        let fog_data = FogUbo {
            color_and_density: [fog_color[0], fog_color[1], fog_color[2], 1.0],
            ambient_and_sun: [ambient_val, 0.0, 1.0, 0.0],
        };
        if let Ok(slice) = self.fog_ubo.mapped_slice_mut() {
            let bytes: &[u8] = bytemuck::bytes_of(&fog_data);
            slice[..bytes.len()].copy_from_slice(bytes);
        }

        // Sky UBO
        let data = SkyUbo {
            horizon: [
                self.sky_horizon[0],
                self.sky_horizon[1],
                self.sky_horizon[2],
                1.0,
            ],
            zenith: [
                self.sky_zenith[0],
                self.sky_zenith[1],
                self.sky_zenith[2],
                1.0,
            ],
            sun_dir: [self.sun_dir[0], self.sun_dir[1], self.sun_dir[2], 0.0],
        };
        if let Ok(slice) = self.sky_ubo.mapped_slice_mut() {
            let bytes: &[u8] = bytemuck::bytes_of(&data);
            slice[..bytes.len()].copy_from_slice(bytes);
        }

        // Shadow UBO (per-frame).
        let shadow_bytes: &[u8] = bytemuck::bytes_of(&self.shadow_ubo_data);
        for frame in self.frames.iter_mut() {
            if let Ok(slice) = frame.shadow_ubo.mapped_slice_mut() {
                slice[..shadow_bytes.len()].copy_from_slice(shadow_bytes);
            }
        }
    }

    /// Upload (or replace) a batch of chunk meshes. Done via a single one-time
    /// staging command buffer for efficiency.
    pub fn upload_chunks(&mut self, uploads: Vec<ChunkUpload>) {
        if uploads.is_empty() {
            return;
        }
        let device = &self.device;
        let alloc = &self.alloc;
        let pool = self.command_pool;
        let queue = self.graphics_queue;

        // Create device-local buffers + staging buffers for each upload.
        struct Pending {
            pos: ChunkPos,
            pass: MeshPass,
            vbo: GpuBuffer,
            ibo: GpuBuffer,
            staging: GpuBuffer,
            v_offset: vk::DeviceSize,
            i_size: vk::DeviceSize,
            index_count: u32,
        }
        let mut pending: Vec<Pending> = Vec::with_capacity(uploads.len());

        // Build one big staging buffer per chunk (vertices then indices packed).
        let cmd = match begin_one_time(device, pool) {
            Ok(c) => c,
            Err(e) => {
                log::error!("begin_one_time failed: {e}");
                return;
            }
        };

        for u in uploads {
            if u.vertices.is_empty() || u.indices.is_empty() {
                continue;
            }
            let v_size = u.vertices.len() as vk::DeviceSize;
            let i_size = u.indices.len() as vk::DeviceSize;
            let staging = match GpuBuffer::host_visible(
                device,
                alloc,
                v_size + i_size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                "chunk_staging",
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("staging alloc failed: {e}");
                    continue;
                }
            };
            let mut staging = staging;
            if let Err(e) = staging.upload(&u.vertices) {
                log::error!("staging vertex upload: {e}");
                staging.destroy(device, alloc);
                continue;
            }
            // Copy indices after vertices in the staging buffer.
            {
                let slice = match staging.mapped_slice_mut() {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("staging map: {e}");
                        staging.destroy(device, alloc);
                        continue;
                    }
                };
                slice[v_size as usize..(v_size + i_size) as usize].copy_from_slice(&u.indices);
            }

            let vbo = match GpuBuffer::device_local(
                device,
                alloc,
                v_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                "chunk_vbo",
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("vbo alloc: {e}");
                    staging.destroy(device, alloc);
                    continue;
                }
            };
            let ibo = match GpuBuffer::device_local(
                device,
                alloc,
                i_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                "chunk_ibo",
            ) {
                Ok(b) => b,
                Err(e) => {
                    log::error!("ibo alloc: {e}");
                    staging.destroy(device, alloc);
                    vbo.destroy(device, alloc);
                    continue;
                }
            };

            // Record copies.
            let v_region = vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(0)
                .size(v_size);
            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, vbo.buffer, &[v_region]);
            }
            let i_region = vk::BufferCopy::default()
                .src_offset(v_size)
                .dst_offset(0)
                .size(i_size);
            unsafe {
                device.cmd_copy_buffer(cmd, staging.buffer, ibo.buffer, &[i_region]);
            }

            pending.push(Pending {
                pos: u.pos,
                pass: u.pass,
                vbo,
                ibo,
                staging,
                v_offset: v_size,
                i_size,
                index_count: u.index_count,
            });
        }

        // Use a per-batch fence instead of queue_wait_idle to avoid stalling
        // the entire GPU queue (which kills frame rate during chunk loading).
        let upload_fence =
            match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
                Ok(f) => f,
                Err(e) => {
                    log::error!("create_fence failed: {e}");
                    return;
                }
            };
        unsafe {
            if let Err(e) = device.end_command_buffer(cmd) {
                log::error!("end_command_buffer (upload) failed: {e:?}");
                device.destroy_fence(upload_fence, None);
                return;
            }
            let command_buffers = [cmd];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            if let Err(e) = device.queue_submit(queue, &[submit_info], upload_fence) {
                log::error!("upload queue_submit failed: {e}");
                device.destroy_fence(upload_fence, None);
                return;
            }
            // Wait only for this batch to finish (not all GPU work).
            if let Err(e) = device.wait_for_fences(&[upload_fence], true, u64::MAX) {
                log::error!("wait_for_fences (upload) failed: {e}");
            }
            device.destroy_fence(upload_fence, None);
            device.free_command_buffers(pool, &[cmd]);
        }

        // Insert into the chunk map. Each chunk can have both an opaque and a
        // transparent pass — store them in the same ChunkBuffers entry rather
        // than overwriting one with the other.
        let mut chunks = self.chunks.write();
        for p in pending {
            let entry = chunks.entry(p.pos).or_insert_with(ChunkBuffers::new);
            let slot = match p.pass {
                MeshPass::Opaque => &mut entry.opaque,
                MeshPass::Transparent => &mut entry.transparent,
            };
            // Replace old buffers for this pass if present.
            if let Some(old) = slot.take() {
                old.vbo.destroy(&self.device, &self.alloc);
                old.ibo.destroy(&self.device, &self.alloc);
            }
            *slot = Some(PassBuffers {
                vbo: p.vbo,
                ibo: p.ibo,
                index_count: p.index_count,
            });
            // Staging can be freed now that the queue is idle.
            p.staging.destroy(&self.device, &self.alloc);
            let _ = (p.v_offset, p.i_size); // already used above
        }
    }

    /// Remove a chunk's GPU buffers (called when the streamer unloads it).
    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        let mut chunks = self.chunks.write();
        if let Some(bufs) = chunks.remove(&pos) {
            bufs.destroy(&self.device, &self.alloc);
        }
    }

    /// Number of chunks currently on the GPU.
    pub fn chunk_count(&self) -> usize {
        self.chunks.read().len()
    }

    /// Toggle wireframe rendering.
    pub fn toggle_wireframe(&mut self) {
        self.wireframe_enabled = !self.wireframe_enabled;
    }

    /// Whether wireframe rendering is active.
    pub fn is_wireframe(&self) -> bool {
        self.wireframe_enabled
    }

    /// Get the active chunk pipeline (fill or wireframe).
    fn active_pipeline(&self) -> vk::Pipeline {
        if self.wireframe_enabled {
            self.wireframe_pipeline
        } else {
            self.pipeline
        }
    }

    /// Render one frame and present it. `camera` drives view-projection + culling.
    /// `ui` provides optional overlay vertices (crosshair, hotbar, pause menu).
    pub fn draw_frame(
        &mut self,
        camera: Camera,
        ui: Option<&UiDrawData>,
        game_time: f32,
        underwater: bool,
    ) -> Result<()> {
        // Resize-check, upload UI, precompute camera matrices. Everything we need
        // before submitting any GPU work.
        let (view_proj, vp_cols, frustum, ui_index_count) = self.prepare_frame(&camera, ui)?;

        let frame_idx = self.frame_counter % FRAMES_IN_FLIGHT;
        // Copy out the per-frame handles (all `Copy`) so we don't hold an
        // immutable borrow of `self.frames` across the mutable UBO update below.
        let (cmd, in_flight_fence, image_available, render_finished, descriptor_set) = {
            let f = &self.frames[frame_idx];
            (
                f.cmd,
                f.in_flight_fence,
                f.image_available,
                f.render_finished,
                f.descriptor_set,
            )
        };

        // Wait for this frame's previous use to finish.
        self.wait_for_fence_reset(in_flight_fence)?;

        // Write UBO data now that the previous frame is done with this slot.
        self.flush_pending_ubos();

        // Read back the previous frame's GPU timestamps (now available after fence wait).
        // On the first 1-2 frames the queries haven't been written yet, so we
        // tolerate VK_NOT_READY and just skip the update.
        {
            let prev_offset = (frame_idx as u32) * GPU_TIMESTAMP_COUNT;
            let mut timestamps = [0u64; GPU_TIMESTAMP_COUNT as usize];
            let read_ok = unsafe {
                self.device.get_query_pool_results(
                    self.query_pool,
                    prev_offset,
                    &mut timestamps,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            if let Ok(()) = read_ok {
                let ns_to_ms = self.timestamp_period / 1_000_000.0;
                let t = &timestamps;
                self.timings = GpuTimings {
                    shadow_ms: (t[1] - t[0]) as f32 * ns_to_ms,
                    sky_ms: (t[2] - t[1]) as f32 * ns_to_ms,
                    opaque_ms: (t[3] - t[2]) as f32 * ns_to_ms,
                    transparent_ms: (t[4] - t[3]) as f32 * ns_to_ms,
                    ui_ms: (t[5] - t[4]) as f32 * ns_to_ms,
                    post_ms: (t[7] - t[6]) as f32 * ns_to_ms,
                    frame_ms: (t[7] - t[0]) as f32 * ns_to_ms,
                };
            }
            // On error (queries not yet available), keep previous timings.
        }

        // Acquire the next swapchain image. NOT_READY / OUT_OF_DATE / SUBOPTIMAL
        // are non-fatal: skip this frame and retry next time.
        let acquire_result = unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        };
        let (image_index, _suboptimal) = match acquire_result {
            Ok(pair) => pair,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                self.needs_resize = true;
                return Ok(());
            }
            Err(vk::Result::NOT_READY) | Err(vk::Result::TIMEOUT) => {
                // No image available right now; try again next frame.
                return Ok(());
            }
            Err(e) => return Err(anyhow!("acquire_next_image: {e:?}")),
        };

        // Update camera UBO.
        self.update_camera_ubo(&camera, underwater, frame_idx)?;

        let device = &self.device;

        // Record command buffer (cmd was copied out above).
        let query_offset = (frame_idx as u32) * GPU_TIMESTAMP_COUNT;
        unsafe {
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            // Reset query pool for this frame's slice and write frame start timestamp.
            device.cmd_reset_query_pool(cmd, self.query_pool, query_offset, GPU_TIMESTAMP_COUNT);
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                self.query_pool,
                query_offset, // timestamp 0: frame start
            );
        }

        // ── Shadow pass: render chunk depth from the light's perspective ──
        self.record_shadow_pass(device, cmd, Some(query_offset + 1));


        self.record_main_pass_setup(device, cmd, image_index, descriptor_set);

        // ── Sky pass: draw the full-screen sky before chunks ──
        self.record_sky_pass(device, cmd, view_proj, camera.pos, descriptor_set, Some(query_offset + 2));

        self.record_chunk_passes(
            device,
            cmd,
            &frustum,
            &vp_cols,
            game_time,
            Some(query_offset + 3),
            Some(query_offset + 4),
        );

        // ── UI overlay pass ──
        if ui_index_count > 0 {
            self.record_ui(device, cmd, ui_index_count);
        }

        unsafe {
            // Timestamp 5: UI end.
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                self.query_pool,
                query_offset + 5,
            );
            device.cmd_end_render_pass(cmd);
        }

        // Timestamp 6: main pass end.
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.query_pool,
                query_offset + 6,
            );
        }

        // ── Post-processing pass: sample offscreen → swapchain ──
        self.record_post_pass(device, cmd, image_index, Some(query_offset + 7));
        unsafe {
            device.end_command_buffer(cmd)?;
        }


        // Submit.
        self.record_submit(
            device,
            cmd,
            &[image_available],
            &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
            &[render_finished],
            in_flight_fence,
        )?;

        // Present.
        self.record_present(image_index, &[render_finished], true)?;

        self.frame_counter += 1;
        Ok(())
    }

    /// Render one frame without presenting and read the colour attachment back
    /// as RGBA8 pixels (used by the engine to save a verification screenshot).
    pub fn capture_frame(
        &mut self,
        camera: Camera,
        ui: Option<&UiDrawData>,
        game_time: f32,
        underwater: bool,
    ) -> Result<Vec<u8>> {
        // Resize-check, upload UI, precompute camera matrices.
        let (view_proj, vp_cols, frustum, ui_index_count) = self.prepare_frame(&camera, ui)?;

        // Wait for frame 0's previous GPU work to complete before we
        // write to its UBO and descriptor set.
        self.wait_for_fence_reset(self.frames[0].in_flight_fence)?;

        // Write UBO data now that the previous frame is done.
        self.flush_pending_ubos();

        let device = &self.device;
        let cmd = begin_one_time(device, self.command_pool)?;

        // Acquire an image. A fence signals when the acquisition is safe; we
        // destroy it as soon as the acquire completes. The submit fence (used
        // later to wait for the readback command buffer) is created just before
        // queue_submit so it only needs cleanup on a narrow window of errors.
        let acquire_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }?;
        let (image_index, _) = match unsafe {
            self.swapchain_device.acquire_next_image(
                self.swapchain,
                u64::MAX,
                vk::Semaphore::null(),
                acquire_fence,
            )
        } {
            Ok(pair) => pair,
            Err(e) => {
                unsafe { device.destroy_fence(acquire_fence, None) };
                return Err(anyhow!("capture acquire: {e:?}"));
            }
        };
        // Wait for the acquisition to complete before using the image.
        unsafe {
            if let Err(e) = device.wait_for_fences(&[acquire_fence], true, u64::MAX) {
                device.destroy_fence(acquire_fence, None);
                return Err(anyhow!("capture wait_for_fences: {e:?}"));
            }
            device.destroy_fence(acquire_fence, None);
        }

        // Update camera UBO (frame 0).
        self.update_camera_ubo(&camera, underwater, 0)?;
        let device = &self.device;

        // ── Shadow pass (capture) ──
        self.record_shadow_pass(device, cmd, None);


        self.record_main_pass_setup(device, cmd, image_index, self.frames[0].descriptor_set);

        // ── Sky pass ──
        self.record_sky_pass(device, cmd, view_proj, camera.pos, self.frames[0].descriptor_set, None);

        self.record_chunk_passes(device, cmd, &frustum, &vp_cols, game_time, None, None);

        // ── UI overlay pass ──
        if ui_index_count > 0 {
            self.record_ui(device, cmd, ui_index_count);
        }

        unsafe {
            device.cmd_end_render_pass(cmd);
        }

        // ── Post pass (capture) ──
        self.record_post_pass(device, cmd, image_index, None);


        // Transition the swapchain image from PRESENT_SRC (the post pass's
        // final layout) to TRANSFER_SRC so we can copy it back to the host.
        transition_image_layout(
            device,
            cmd,
            self.swapchain_images[image_index as usize],
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );

        let extent = self.swapchain_extent;
        let pixel_count = (extent.width * extent.height) as usize;
        let row_pitch = extent.width * 4; // RGBA8
        let buf_size = (row_pitch * extent.height) as vk::DeviceSize;
        let mut readback = GpuBuffer::host_visible(
            device,
            &self.alloc,
            buf_size,
            vk::BufferUsageFlags::TRANSFER_DST,
            "capture_readback",
        )?;

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(extent.width)
            .buffer_image_height(extent.height)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            });
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                self.swapchain_images[image_index as usize],
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.buffer,
                &[region],
            );
            device.end_command_buffer(cmd)?;
        }

        let submit_fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(|e| anyhow!("capture submit_fence: {e:?}"))?;
        self.record_submit(device, cmd, &[], &[], &[], submit_fence)?;
        unsafe {
            if let Err(e) = device.wait_for_fences(&[submit_fence], true, u64::MAX) {
                device.destroy_fence(submit_fence, None);
                return Err(anyhow!("capture wait_for_fences (submit): {e:?}"));
            }
            device.destroy_fence(submit_fence, None);
            device.free_command_buffers(self.command_pool, &[cmd]);
        }

        let slice = readback.mapped_slice_mut()?;
        let mut out = vec![0u8; pixel_count * 4];
        out.copy_from_slice(&slice[..pixel_count * 4]);
        readback.destroy(device, &self.alloc);

        // Transition the image back to PRESENT_SRC and present it so the
        // swapchain image is returned to the pool (otherwise repeated captures
        // leak images until acquire_next_image deadlocks/crashes).
        let present_cmd = begin_one_time(device, self.command_pool)?;
        transition_image_layout(
            device,
            present_cmd,
            self.swapchain_images[image_index as usize],
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageAspectFlags::COLOR,
            1,
            1,
        );
        unsafe {
            device.end_command_buffer(present_cmd)?;
        }

        let present_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }?;
        let command_buffers = [present_cmd];
        let signal_semaphores = [present_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);
        let submit_infos = [submit_info];
        unsafe {
            device.queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())?;
        }

        // `queue_submit` was the last OLD `device` use; the OLD `&self.device`
        // borrow ends here. `record_present` takes `&mut self` next.
        self.record_present(image_index, &signal_semaphores, false)?;

        // Fresh borrow for the post-present cleanup (record_present held
        // &mut self, so we need an immutable re-acquire here).
        let device = &self.device;

        unsafe {
            device.device_wait_idle()?;
            device.destroy_semaphore(present_semaphore, None);
            device.free_command_buffers(self.command_pool, &[present_cmd]);
        }

        // The swapchain is typically B8G8R8A8 (sRGB or unorm); the readback is
        // in that channel order. Convert to RGBA8 so callers can save it directly.
        let is_bgra = matches!(
            self.swapchain_format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
        );
        if is_bgra {
            for px in out.chunks_exact_mut(4) {
                px.swap(0, 2); // B <-> R
            }
        }

        // Keep frame_counter in sync so the next draw_frame picks the right slot.
        self.frame_counter += 1;

        Ok(out)
    }

    /// Record both chunk draw passes (opaque then transparent) into `cmd`.
    ///
    /// Both `draw_frame` and `capture_frame` need identical chunk rendering;
    /// this is the shared implementation. Uses the **collect-then-drop** lock
    /// pattern: acquire `self.chunks.read()` once to build the list of
    /// visible chunks, release it before recording draws. This keeps chunk
    /// uploads (`upload_chunks` taking `self.chunks.write()`) from blocking
    /// for the duration of push-constant filling + draw command recording.
    ///
    /// `opaque_end_timestamp_query` and `transparent_end_timestamp_query`
    /// are `Some(query_offset + 3)` and `Some(query_offset + 4)` from
    /// `draw_frame` (GPU profiling); `capture_frame` passes `None` for both
    /// since it doesn't run the timestamp pool.
    ///
    /// `vp_cols` is the 16-float view-projection matrix in column-major order.
    fn record_chunk_passes(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        frustum: &Frustum,
        vp_cols: &[f32],
        game_time: f32,
        opaque_end_timestamp_query: Option<u32>,
        transparent_end_timestamp_query: Option<u32>,
    ) {
        // Collect visible chunk buffer handles under the read lock, then
        // drop the lock so `upload_chunks` can proceed while we record.
        let (opaque_draws, transparent_draws): (
            Vec<(ChunkPos, vk::Buffer, vk::Buffer, u32)>,
            Vec<(ChunkPos, vk::Buffer, vk::Buffer, u32)>,
        ) = {
            let chunks = self.chunks.read();
            let mut opaque = Vec::new();
            let mut transparent = Vec::new();
            for (&pos, bufs) in chunks.iter() {
                let origin = chunk_origin(pos);
                let min =
                    Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
                let max = min + Vec3::splat(voxel_core::CHUNK_SIZE as f32);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                if let Some(b) = &bufs.opaque {
                    opaque.push((pos, b.vbo.buffer, b.ibo.buffer, b.index_count));
                }
                if let Some(b) = &bufs.transparent {
                    transparent.push((pos, b.vbo.buffer, b.ibo.buffer, b.index_count));
                }
            }
            (opaque, transparent)
        };

        let issue = |cmd: vk::CommandBuffer,
                     draws: &[(ChunkPos, vk::Buffer, vk::Buffer, u32)]| {
            for &(pos, vbo_buf, ibo_buf, index_count) in draws {
                let origin = chunk_origin(pos);
                let mut push = [0f32; 24];
                push[0] = origin.x as f32;
                push[1] = origin.y as f32;
                push[2] = origin.z as f32;
                push[4..20].copy_from_slice(vp_cols);
                push[20] = game_time;
                unsafe {
                    device.cmd_push_constants(
                        cmd,
                        self.pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::bytes_of(&push),
                    );
                    let vbo = [vbo_buf];
                    device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                    device.cmd_bind_index_buffer(
                        cmd,
                        ibo_buf,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
                }
            }
        };

        issue(cmd, &opaque_draws);
        unsafe {
            if let Some(q) = opaque_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }
            // Switch to the transparent pipeline (no culling) so water is
            // visible from both sides.
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.transparent_pipeline,
            );
            issue(cmd, &transparent_draws);
            if let Some(q) = transparent_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }
        }
    }

    /// Record the cascaded shadow pass into `cmd`. Both `draw_frame` and
    /// `capture_frame` need identical shadow rendering; this is the shared
    /// implementation. If `shadow_end_timestamp_query` is `Some`, a
    /// `cmd_write_timestamp` is emitted at the end of the pass (used by
    /// the GPU profiling path in `draw_frame`; capture skips it).
    fn record_shadow_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        shadow_end_timestamp_query: Option<u32>,
    ) {
        let shadow_extent = vk::Extent2D {
            width: 2048,
            height: 2048,
        };
        let shadow_vp = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(2048.0)
            .height(2048.0)
            .min_depth(0.0)
            .max_depth(1.0);
        let shadow_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: shadow_extent,
        };
        let chunks = self.chunks.read();
        for cascade in 0..4u32 {
            let clear_values = [vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            }];
            let shadow_begin = vk::RenderPassBeginInfo::default()
                .render_pass(self.shadow_render_pass)
                .framebuffer(self.shadow_framebuffers[cascade as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: shadow_extent,
                })
                .clear_values(&clear_values);
            unsafe {
                device.cmd_begin_render_pass(
                    cmd,
                    &shadow_begin,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_set_viewport(cmd, 0, &[shadow_vp]);
                device.cmd_set_scissor(cmd, 0, &[shadow_scissor]);
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.shadow_pipeline,
                );

                let vp = self.shadow_ubo_data.cascade_vps[cascade as usize];
                for (&pos, b) in chunks.iter() {
                    let Some(opaque) = &b.opaque else { continue };
                    let origin = chunk_origin(pos);
                    let mut push = [0f32; 20];
                    push[..16].copy_from_slice(&vp);
                    push[16] = origin.x as f32;
                    push[17] = origin.y as f32;
                    push[18] = origin.z as f32;
                    push[19] = 0.0;
                    device.cmd_push_constants(
                        cmd,
                        self.shadow_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytemuck::cast_slice(&push),
                    );
                    let vbo = [opaque.vbo.buffer];
                    device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
                    device.cmd_bind_index_buffer(
                        cmd,
                        opaque.ibo.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );
                    device.cmd_draw_indexed(cmd, opaque.index_count, 1, 0, 0, 0);
                }
                device.cmd_end_render_pass(cmd);
            }
        }
        drop(chunks);

        if let Some(q) = shadow_end_timestamp_query {
            unsafe {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    q,
                );
            }
        }
    }

    /// Record the sky pass (full-screen gradient) into `cmd` and then
    /// re-bind the chunk graphics pipeline + descriptor set so the
    /// subsequent chunk draws see the right state.
    ///
    /// Both `draw_frame` and `capture_frame` need identical sky rendering;
    /// this is the shared implementation. If `sky_end_timestamp_query` is
    /// `Some`, a `cmd_write_timestamp` is emitted on `COLOR_ATTACHMENT_OUTPUT`
    /// after the sky draw (used by the GPU profiling path in `draw_frame`;
    /// capture skips it).
    ///
    /// `chunk_descriptor_set` is restored after the sky draw so the chunk
    /// pass binds to the right per-frame resources.
    fn record_sky_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        view_proj: Mat4,
        camera_pos: Vec3,
        chunk_descriptor_set: vk::DescriptorSet,
        sky_end_timestamp_query: Option<u32>,
    ) {
        let vp = vk::Viewport::default()
            .x(0.0)
            .y(self.swapchain_extent.height as f32)
            .width(self.swapchain_extent.width as f32)
            .height(-(self.swapchain_extent.height as f32))
            .min_depth(0.0)
            .max_depth(1.0);
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };

        // Compute inverse view-projection for the sky shader.
        let inv_vp = if view_proj.determinant().abs() > 1e-10 {
            view_proj.inverse()
        } else {
            Mat4::IDENTITY
        };
        let inv_vp_cols = inv_vp.to_cols_array();
        // Pack inverse VP (64 bytes) + camera position (16 bytes) = 80 bytes.
        let mut sky_push_data = [0.0f32; 20];
        sky_push_data[..16].copy_from_slice(&inv_vp_cols);
        sky_push_data[16] = camera_pos.x;
        sky_push_data[17] = camera_pos.y;
        sky_push_data[18] = camera_pos.z;
        sky_push_data[19] = 0.0;

        let sky_desc_sets = [self.sky_descriptor_set];

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[vp]);
            device.cmd_set_scissor(cmd, 0, &[scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.sky_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.sky_pipeline_layout,
                0,
                &sky_desc_sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.sky_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::cast_slice(&sky_push_data),
            );
            // Draw 3 vertices = full-screen triangle (no vertex buffer).
            device.cmd_draw(cmd, 3, 1, 0, 0);
            if let Some(q) = sky_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    self.query_pool,
                    q,
                );
            }

            // Restore the chunk pipeline + descriptor set so chunk draws
            // see the right state. The viewport is already correct (same
            // negative-height viewport) so it doesn't need to be reset.
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.active_pipeline(),
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[chunk_descriptor_set],
                &[],
            );
        }
    }

    /// Record the post-processing pass into `cmd`. Both `draw_frame` and
    /// `capture_frame` need the same offscreen → swapchain blit; this is
    /// the shared implementation.
    ///
    /// `post_end_timestamp_query` is `Some(query_offset + 7)` from
    /// `draw_frame` (GPU profiling). Capture frames pass `None` because
    /// they don't run the timing pool.
    fn record_post_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        post_end_timestamp_query: Option<u32>,
    ) {
        let post_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.post_render_pass)
            .framebuffer(self.post_framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain_extent,
            })
            .clear_values(&[]);
        let post_vp = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let post_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        unsafe {
            device.cmd_begin_render_pass(cmd, &post_begin, vk::SubpassContents::INLINE);
            device.cmd_set_viewport(cmd, 0, &[post_vp]);
            device.cmd_set_scissor(cmd, 0, &[post_scissor]);
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.post_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.post_pipeline_layout,
                0,
                &[self.post_descriptor_sets[image_index as usize]],
                &[],
            );
            let post_push = self.post_params;
            device.cmd_push_constants(
                cmd,
                self.post_pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::cast_slice(&post_push),
            );
            device.cmd_draw(cmd, 3, 1, 0, 0);
            device.cmd_end_render_pass(cmd);
            if let Some(q) = post_end_timestamp_query {
                device.cmd_write_timestamp(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    self.query_pool,
                    q,
                );
            }
        }
    }

    /// Begin the offscreen render pass for `image_index`, bind the active chunk
    /// pipeline, and bind the chunk descriptor set. Both `draw_frame` and
    /// `capture_frame` call this with the appropriate descriptor set
    /// (`draw_frame` uses the per-frame set; `capture_frame` uses frame 0).
    /// The render area + clear values are derived from `self.swapchain_extent`
    /// and `self.config.clear_color` internally so callers don't have to pass
    /// the same literal block each time.
    fn record_main_pass_setup(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: u32,
        descriptor_set: vk::DescriptorSet,
    ) {
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: self.config.clear_color,
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.render_pass)
            .framebuffer(self.offscreen_framebuffers[image_index as usize])
            .render_area(render_area)
            .clear_values(&clear_values);
        unsafe {
            device.cmd_begin_render_pass(cmd, &render_pass_begin, vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.active_pipeline());
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
        }
    }

    /// Update `frames[frame_index].camera_ubo` with the current `camera` position
    /// and the underwater-aware fog distance. Required to be called after the
    /// frame's previous GPU work has finished (e.g. after `wait_for_fences`), so
    /// writes don't race GPU reads. Both `draw_frame` and `capture_frame` call
    /// this; only the `frame_index` differs (per-frame in draw, frame 0 in capture).
    fn update_camera_ubo(
        &mut self,
        camera: &Camera,
        underwater: bool,
        frame_index: usize,
    ) -> Result<()> {
        let mut cam_ubo = CameraUbo::default();
        let fog_dist = if underwater {
            self.config.fog_distance * 0.05 // Very close fog underwater for Minecraft-like look.
        } else {
            self.config.fog_distance
        };
        cam_ubo.cam_pos_and_maxdist = [camera.pos.x, camera.pos.y, camera.pos.z, fog_dist];
        let frame = &mut self.frames[frame_index];
        let slice = frame.camera_ubo.mapped_slice_mut()?;
        let bytes: &[u8] = bytemuck::bytes_of(&cam_ubo);
        slice[..bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    /// Common preamble for both `draw_frame` and `capture_frame`: handle a
    /// pending swapchain resize, upload any UI data, and precompute the
    /// camera matrices used by all `record_*_pass` helpers. Returns every
    /// derived value callers need before submitting any GPU work.
    ///
    /// **Note:** `view_proj`/`vp_cols`/`frustum` are computed and returned
    /// BEFORE `update_camera_ubo` is called later in each method. This is
    /// safe because `view_projection` is a pure function of `camera` and
    /// does not read `self` state.
    ///
    /// **Return-tuple order:** `(Mat4, [f32; 16], Frustum, u32)` —
    /// `draw_frame` and `capture_frame` destructure into
    /// `(view_proj, vp_cols, frustum, ui_index_count)`. Keep this order in
    /// sync with both call sites if you ever add a return field.
    fn prepare_frame(
        &mut self,
        camera: &Camera,
        ui: Option<&UiDrawData>,
    ) -> Result<(Mat4, [f32; 16], Frustum, u32)> {
        if self.needs_resize {
            self.recreate_swapchain()?;
            self.needs_resize = false;
        }
        let ui_index_count = ui.map(|u| self.upload_ui(u)).unwrap_or(0);
        let view_proj = camera.view_projection();
        let vp_cols = view_proj.to_cols_array(); // 16 floats, column-major
        let frustum = Frustum::from_view_projection(view_proj);
        Ok((view_proj, vp_cols, frustum, ui_index_count))
    }


    /// Wait for `fence` to signal (the previous frame's GPU work using this
    /// fence is done) and reset it back to unsignalled state. Called once per
    /// frame at the start of both `draw_frame` (uses the per-frame
    /// `in_flight_fence`) and `capture_frame` (uses `self.frames[0].in_flight_fence`).
    fn wait_for_fence_reset(&self, fence: vk::Fence) -> Result<()> {
        unsafe {
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;
            self.device.reset_fences(&[fence])?;
        }
        Ok(())
    }

    /// Build a `vk::SubmitInfo` from the given command buffer + wait/signal
    /// semaphores and submit it to `self.graphics_queue`. Used by both
    /// `draw_frame` and `capture_frame`. The caller owns `fence` (passing
    /// `vk::Fence::null()` is fine if no completion signal is needed).
    fn record_submit(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        fence: vk::Fence,
    ) -> Result<()> {
        let command_buffers = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);
        let submit_infos = [submit_info];
        unsafe {
            device.queue_submit(self.graphics_queue, &submit_infos, fence)?;
        }
        Ok(())
    }

    /// Submit the present request for `image_index` and handle the result.
    /// If `set_resize_on_out_of_date` is true and the result is
    /// `ERROR_OUT_OF_DATE_KHR` or `SUBOPTIMAL_KHR`, `self.needs_resize` is
    /// set so the next frame triggers a swapchain recreate. If false (used by
    /// `capture_frame`), those errors are silently ignored.
    fn record_present(
        &mut self,
        image_index: u32,
        wait_semaphores: &[vk::Semaphore],
        set_resize_on_out_of_date: bool,
    ) -> Result<()> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let result = unsafe {
            self.swapchain_device
                .queue_present(self.present_queue, &present_info)
        };
        match result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR)
                if set_resize_on_out_of_date =>
            {
                self.needs_resize = true;
            }
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                // capture_frame: silently ignore OUT_OF_DATE/SUBOPTIMAL.
            }
            Err(e) => return Err(anyhow!("queue_present: {e:?}")),
        }
        Ok(())
    }

    /// Recreate the swapchain + dependent resources (called on resize).
    fn recreate_swapchain(&mut self) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;
        }
        // Clean up old swapchain-dependent objects.
        for &fb in &self.offscreen_framebuffers {
            unsafe { self.device.destroy_framebuffer(fb, None) };
        }
        self.offscreen_framebuffers.clear();
        for img in self.offscreen_images.drain(..) {
            img.destroy(&self.device, &self.alloc);
        }
        for &fb in &self.post_framebuffers {
            unsafe { self.device.destroy_framebuffer(fb, None) };
        }
        self.post_framebuffers.clear();
        for &v in &self.swapchain_image_views {
            unsafe { self.device.destroy_image_view(v, None) };
        }
        if let Some(depth) = self.depth.take() {
            depth.destroy(&self.device, &self.alloc);
        }
        unsafe {
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
        }

        let (swapchain, swapchain_images, swapchain_format, swapchain_extent) = create_swapchain(
            &self.device,
            &self.swapchain_device,
            &self.surface_instance,
            self.physical_device,
            self.surface,
            self.config.vsync,
        )?;
        let swapchain_image_views =
            create_image_views(&self.device, &swapchain_images, swapchain_format)?;
        let depth_format = find_depth_format(&self.instance, self.physical_device);
        let depth = GpuImage::depth(&self.device, &self.alloc, swapchain_extent, depth_format)?;

        // Recreate offscreen images + framebuffers.
        let mut offscreen_images = Vec::with_capacity(swapchain_images.len());
        for _ in 0..swapchain_images.len() {
            let img = GpuImage::color_attachment(
                &self.device,
                &self.alloc,
                swapchain_extent,
                self.swapchain_format,
                "offscreen",
            )?;
            let cmd_init = begin_one_time(&self.device, self.command_pool)?;
            transition_image_layout(
                &self.device,
                cmd_init,
                img.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                1,
                1,
            );
            end_and_submit(
                &self.device,
                self.command_pool,
                self.graphics_queue,
                cmd_init,
            )?;
            offscreen_images.push(img);
        }
        let offscreen_framebuffers = offscreen_images
            .iter()
            .map(|img| {
                create_framebuffer_with(
                    &self.device,
                    self.render_pass,
                    &[img.view, depth.view],
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        // Recreate post framebuffers (one per swapchain image view).
        let post_framebuffers = swapchain_image_views
            .iter()
            .map(|&view| {
                create_framebuffer_with(
                    &self.device,
                    self.post_render_pass,
                    &[view],
                    swapchain_extent,
                )
            })
            .collect::<Result<Vec<_>>>()?;

        self.swapchain = swapchain;
        self.swapchain_images = swapchain_images;
        self.swapchain_image_views = swapchain_image_views;
        self.swapchain_format = swapchain_format;
        self.swapchain_extent = swapchain_extent;
        self.depth = Some(depth);
        self.offscreen_images = offscreen_images;
        self.offscreen_framebuffers = offscreen_framebuffers;
        self.post_framebuffers = post_framebuffers;
        Ok(())
    }

    /// Record UI draw commands into `cmd`. Must be called inside the render pass
    /// (after the chunk pass). Uploads vertices to the persistent host-visible
    /// buffer, binds the UI pipeline, and draws.
    /// Upload UI vertex/index data to the persistent mapped buffers. Call before
    /// the render pass begins. Returns the index count to draw, or 0 if skipped.
    fn upload_ui(&mut self, ui: &UiDrawData) -> u32 {
        let vbytes = bytemuck::cast_slice(&ui.vertices);
        let ibytes = bytemuck::cast_slice(&ui.indices);
        {
            let vslice = match self.ui_vbo.mapped_slice_mut() {
                Ok(s) => s,
                Err(e) => {
                    log::error!("ui vbo map: {e}");
                    return 0;
                }
            };
            if vbytes.len() > vslice.len() {
                log::warn!(
                    "UI vertices {} exceed buffer {}",
                    vbytes.len(),
                    vslice.len()
                );
                return 0;
            }
            vslice[..vbytes.len()].copy_from_slice(vbytes);
        }
        {
            let islice = match self.ui_ibo.mapped_slice_mut() {
                Ok(s) => s,
                Err(e) => {
                    log::error!("ui ibo map: {e}");
                    return 0;
                }
            };
            if ibytes.len() > islice.len() {
                log::warn!("UI indices {} exceed buffer {}", ibytes.len(), islice.len());
                return 0;
            }
            islice[..ibytes.len()].copy_from_slice(ibytes);
        }
        ui.indices.len() as u32
    }

    /// Record UI draw commands into `cmd`. Must be called inside the render pass.
    fn record_ui(&self, device: &ash::Device, cmd: vk::CommandBuffer, index_count: u32) {
        let ui_viewport = vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(self.swapchain_extent.width as f32)
            .height(self.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);
        let ui_scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.swapchain_extent,
        };
        let push = [
            self.swapchain_extent.width as f32,
            self.swapchain_extent.height as f32,
        ];
        let ui_desc_sets = [self.ui_descriptor_set];
        let vbo = [self.ui_vbo.buffer];

        unsafe {
            device.cmd_set_viewport(cmd, 0, &[ui_viewport]);
            device.cmd_set_scissor(cmd, 0, &[ui_scissor]);
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.ui_pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.ui_pipeline_layout,
                0,
                &ui_desc_sets,
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.ui_pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_bind_vertex_buffers(cmd, 0, &vbo, &[0]);
            device.cmd_bind_index_buffer(cmd, self.ui_ibo.buffer, 0, vk::IndexType::UINT32);
            device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {
            let _ = device.device_wait_idle();
        }

        // Chunk buffers.
        let chunks = std::mem::take(&mut self.chunks).into_inner();
        for (_, bufs) in chunks {
            bufs.destroy(device, &self.alloc);
        }

        // Per-frame.
        for f in self.frames.drain(..) {
            unsafe {
                device.destroy_semaphore(f.image_available, None);
                device.destroy_semaphore(f.render_finished, None);
                device.destroy_fence(f.in_flight_fence, None);
            }
            f.camera_ubo.destroy(device, &self.alloc);
            f.shadow_ubo.destroy(device, &self.alloc);
        }

        // Atlas + fog UBO + UI resources + sky resources.
        self.atlas.destroy_in_place(device, &self.alloc);
        self.fog_ubo.destroy_in_place(device, &self.alloc);
        self.font_texture.destroy_in_place(device, &self.alloc);
        self.ui_vbo.destroy_in_place(device, &self.alloc);
        self.ui_ibo.destroy_in_place(device, &self.alloc);
        self.sky_ubo.destroy_in_place(device, &self.alloc);

        // Shadow resources.
        for &fb in &self.shadow_framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        self.shadow_framebuffers.clear();
        for &v in &self.shadow_layer_views {
            unsafe { device.destroy_image_view(v, None) };
        }
        self.shadow_layer_views.clear();
        self.shadow_image.destroy_in_place(device, &self.alloc);

        // Offscreen resources.
        for &fb in &self.offscreen_framebuffers {
            unsafe { device.destroy_framebuffer(fb, None) };
        }
        self.offscreen_framebuffers.clear();
        for img in self.offscreen_images.drain(..) {
            img.destroy(device, &self.alloc);
        }

        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_query_pool(self.query_pool, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline(self.wireframe_pipeline, None);
            device.destroy_pipeline(self.transparent_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_pipeline(self.ui_pipeline, None);
            device.destroy_pipeline_layout(self.ui_pipeline_layout, None);
            device.destroy_descriptor_pool(self.ui_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.ui_descriptor_set_layout, None);
            device.destroy_pipeline(self.sky_pipeline, None);
            device.destroy_pipeline_layout(self.sky_pipeline_layout, None);
            device.destroy_descriptor_pool(self.sky_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.sky_descriptor_set_layout, None);
            device.destroy_render_pass(self.render_pass, None);
            // Shadow pass resources.
            device.destroy_sampler(self.shadow_sampler, None);
            device.destroy_pipeline(self.shadow_pipeline, None);
            device.destroy_pipeline_layout(self.shadow_pipeline_layout, None);
            device.destroy_render_pass(self.shadow_render_pass, None);
            // Post pass resources.
            for &fb in &self.post_framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            self.post_framebuffers.clear();
            device.destroy_sampler(self.post_sampler, None);
            device.destroy_pipeline(self.post_pipeline, None);
            device.destroy_pipeline_layout(self.post_pipeline_layout, None);
            device.destroy_descriptor_pool(self.post_descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.post_descriptor_set_layout, None);
            device.destroy_render_pass(self.post_render_pass, None);
            if let Some(depth) = self.depth.take() {
                depth.destroy(device, &self.alloc);
            }
            for &v in &self.swapchain_image_views {
                device.destroy_image_view(v, None);
            }
            self.swapchain_device
                .destroy_swapchain(self.swapchain, None);
            device.destroy_command_pool(self.command_pool, None);
        }

        // Drop the allocator (frees remaining allocations) BEFORE destroying the device.
        unsafe {
            ManuallyDrop::drop(&mut self.alloc);
        }

        // Device, surface, debug messenger, instance.
        unsafe {
            device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
            if let Some(m) = self.debug_messenger.take() {
                let du = ash::ext::debug_utils::Instance::new(&self._entry, &self.instance);
                du.destroy_debug_utils_messenger(m, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// --- helpers --------------------------------------------------------------

fn create_instance(
    entry: &Entry,
    display: RawDisplayHandle,
    validation: bool,
) -> Result<AshInstance> {
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"voxel")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"voxel-engine")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 3, 0));

    let surface_exts = ash_window::enumerate_required_extensions(display)
        .map_err(|e| anyhow!("enumerate_required_extensions: {e:?}"))?;
    let mut extension_names: Vec<*const c_char> = surface_exts.to_vec();
    if validation {
        extension_names.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
    }

    let layers: Vec<&CStr> = if validation {
        vec![c"VK_LAYER_KHRONOS_validation"]
    } else {
        vec![]
    };
    let layer_ptrs: Vec<*const c_char> = layers.iter().map(|l| l.as_ptr()).collect();

    let mut create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    create_info = create_info
        .enabled_extension_names(&extension_names)
        .enabled_layer_names(&layer_ptrs);

    unsafe { entry.create_instance(&create_info, None) }
        .map_err(|e| anyhow!("create_instance: {e:?}"))
}

unsafe extern "system" fn debug_callback(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !data.is_null() {
        let data = &*data;
        let msg = unsafe { CStr::from_ptr(data.p_message) };
        log::warn!("[validation] {}", msg.to_string_lossy());
    }
    vk::FALSE
}

fn create_debug_messenger(
    entry: &Entry,
    instance: &AshInstance,
) -> Result<vk::DebugUtilsMessengerEXT> {
    let du = ash::ext::debug_utils::Instance::new(entry, instance);
    let severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    let ty = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE;
    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(severity)
        .message_type(ty)
        .pfn_user_callback(Some(debug_callback));
    unsafe { du.create_debug_utils_messenger(&create_info, None) }
        .map_err(|e| anyhow!("create_debug_utils_messenger: {e:?}"))
}

fn pick_physical_device(
    instance: &AshInstance,
    surface: &ash::khr::surface::Instance,
    actual_surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, QueueFamilies)> {
    let physicals = unsafe { instance.enumerate_physical_devices() }
        .map_err(|e| anyhow!("enumerate_physical_devices: {e:?}"))?;
    for pdev in physicals {
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        if props.device_type == vk::PhysicalDeviceType::CPU {
            continue;
        }
        if let Some(q) = find_queue_families(instance, surface, pdev, actual_surface) {
            return Ok((pdev, q));
        }
    }
    Err(anyhow!("no suitable Vulkan physical device found"))
}

fn find_queue_families(
    instance: &AshInstance,
    surface: &ash::khr::surface::Instance,
    pdev: vk::PhysicalDevice,
    actual_surface: vk::SurfaceKHR,
) -> Option<QueueFamilies> {
    let props = unsafe { instance.get_physical_device_queue_family_properties(pdev) };
    let mut graphics = None;
    let mut present = None;
    for (i, q) in props.iter().enumerate() {
        if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) && graphics.is_none() {
            graphics = Some(i as u32);
        }
        let supports =
            unsafe { surface.get_physical_device_surface_support(pdev, i as u32, actual_surface) }
                .unwrap_or(false);
        if supports && present.is_none() {
            present = Some(i as u32);
        }
    }
    Some(QueueFamilies {
        graphics: graphics?,
        present: present?,
    })
}

fn create_logical_device(
    instance: &AshInstance,
    pdev: vk::PhysicalDevice,
    queues: QueueFamilies,
    _surface: &ash::khr::surface::Instance,
    _actual_surface: vk::SurfaceKHR,
) -> Result<(ash::Device, vk::Queue, vk::Queue)> {
    let mut unique = vec![queues.graphics, queues.present];
    unique.sort();
    unique.dedup();
    let priorities = [1.0f32];
    let queue_infos: Vec<vk::DeviceQueueCreateInfo> = unique
        .iter()
        .map(|&q| {
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(q)
                .queue_priorities(&priorities)
        })
        .collect();

    let extension_names = [vk::KHR_SWAPCHAIN_NAME.as_ptr()];
    let features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(false);
    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extension_names)
        .enabled_features(&features);

    let device = unsafe { instance.create_device(pdev, &create_info, None) }
        .map_err(|e| anyhow!("create_device: {e:?}"))?;
    let graphics_queue = unsafe { device.get_device_queue(queues.graphics, 0) };
    let present_queue = unsafe { device.get_device_queue(queues.present, 0) };
    Ok((device, graphics_queue, present_queue))
}

fn create_swapchain(
    _device: &ash::Device,
    swapchain_device: &ash::khr::swapchain::Device,
    surface: &ash::khr::surface::Instance,
    pdev: vk::PhysicalDevice,
    actual_surface: vk::SurfaceKHR,
    vsync: bool,
) -> Result<(vk::SwapchainKHR, Vec<vk::Image>, vk::Format, vk::Extent2D)> {
    let caps = unsafe { surface.get_physical_device_surface_capabilities(pdev, actual_surface) }
        .map_err(|e| anyhow!("surface capabilities: {e:?}"))?;
    let formats = unsafe { surface.get_physical_device_surface_formats(pdev, actual_surface) }
        .map_err(|e| anyhow!("surface formats: {e:?}"))?;
    let present_modes =
        unsafe { surface.get_physical_device_surface_present_modes(pdev, actual_surface) }
            .map_err(|e| anyhow!("surface present modes: {e:?}"))?;

    let format = formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .or_else(|| formats.first())
        .ok_or_else(|| anyhow!("no surface formats"))?;

    let present_mode = if vsync {
        vk::PresentModeKHR::FIFO
    } else {
        present_modes
            .iter()
            .copied()
            .find(|m| *m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO)
    };

    let extent = if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: 1280,
            height: 720,
        }
    };
    let mut image_count = caps.min_image_count + 1;
    if caps.max_image_count > 0 && image_count > caps.max_image_count {
        image_count = caps.max_image_count;
    }

    let create_info = vk::SwapchainCreateInfoKHR::default()
        .surface(actual_surface)
        .min_image_count(image_count)
        .image_format(format.format)
        .image_color_space(format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true);

    let swapchain = unsafe { swapchain_device.create_swapchain(&create_info, None) }
        .map_err(|e| anyhow!("create_swapchain: {e:?}"))?;
    let images = unsafe { swapchain_device.get_swapchain_images(swapchain) }
        .map_err(|e| anyhow!("get_swapchain_images: {e:?}"))?;
    Ok((swapchain, images, format.format, extent))
}

fn create_image_views(
    device: &ash::Device,
    images: &[vk::Image],
    format: vk::Format,
) -> Result<Vec<vk::ImageView>> {
    let mut views = Vec::with_capacity(images.len());
    for &img in images {
        views.push(create_image_view(
            device,
            img,
            format,
            vk::ImageAspectFlags::COLOR,
        )?);
    }
    Ok(views)
}

fn find_depth_format(instance: &AshInstance, pdev: vk::PhysicalDevice) -> vk::Format {
    for &f in &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ] {
        let props = unsafe { instance.get_physical_device_format_properties(pdev, f) };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return f;
        }
    }
    vk::Format::D32_SFLOAT
}

fn create_render_pass(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: vk::Format,
) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_ref = vk::AttachmentReference::default()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs)
        .depth_stencil_attachment(&depth_ref);

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let attachments = [color_attachment, depth_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_render_pass: {e:?}"))
}

fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_descriptor_set_layout: {e:?}"))
}

fn create_descriptor_pool(device: &ash::Device, max_sets: usize) -> Result<vk::DescriptorPool> {
    let pool_sizes = [
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: (max_sets * 3) as u32,
        },
        vk::DescriptorPoolSize {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: (max_sets * 2) as u32,
        },
    ];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(max_sets as u32)
        .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
    unsafe { device.create_descriptor_pool(&create_info, None) }
        .map_err(|e| anyhow!("create_descriptor_pool: {e:?}"))
}

fn allocate_descriptor_sets(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    count: usize,
) -> Result<Vec<vk::DescriptorSet>> {
    let layouts = vec![layout; count];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| anyhow!("allocate_descriptor_sets: {e:?}"))
}

fn update_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    camera_buffer: vk::Buffer,
    fog_buffer: vk::Buffer,
    atlas_view: vk::ImageView,
    atlas_sampler: vk::Sampler,
    shadow_view: vk::ImageView,
    shadow_sampler: vk::Sampler,
    shadow_buffer: vk::Buffer,
) {
    let cam_info = vk::DescriptorBufferInfo::default()
        .buffer(camera_buffer)
        .offset(0)
        .range(std::mem::size_of::<CameraUbo>() as u64);
    let fog_info = vk::DescriptorBufferInfo::default()
        .buffer(fog_buffer)
        .offset(0)
        .range(std::mem::size_of::<FogUbo>() as u64);
    let shadow_info = vk::DescriptorBufferInfo::default()
        .buffer(shadow_buffer)
        .offset(0)
        .range(std::mem::size_of::<ShadowUbo>() as u64);
    let atlas_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(atlas_view)
        .sampler(atlas_sampler);
    let shadow_img_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(shadow_view)
        .sampler(shadow_sampler);

    let cam_infos = [cam_info];
    let atlas_infos = [atlas_info];
    let fog_infos = [fog_info];
    let shadow_img_infos = [shadow_img_info];
    let shadow_buf_infos = [shadow_info];

    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&cam_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&atlas_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&fog_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&shadow_img_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&shadow_buf_infos),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn create_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(96); // vec4 + mat4 + vec4 (origin, view_proj, time)
    let set_layouts = [set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_pipeline_layout: {e:?}"))
}

fn create_graphics_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    polygon_mode: vk::PolygonMode,
    cull_mode: vk::CullModeFlags,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/chunk.frag.spv"));
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("vert shader module: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("frag shader module: {e:?}"))?;

    let shadow_enabled = [0u32];
    let spec_map = [vk::SpecializationMapEntry::default()
        .constant_id(0)
        .size(std::mem::size_of::<u32>())
        .offset(0)];
    let spec_info = vk::SpecializationInfo::default()
        .map_entries(&spec_map)
        .data(bytemuck::cast_slice(&shadow_enabled));
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main")
            .specialization_info(&spec_info),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .cull_mode(cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines =
        result.map_err(|(_pipelines, e)| anyhow!("create_graphics_pipelines: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── UI pipeline helpers ──────────────────────────────────────────────────

/// Convert SPIR-V bytes (from `include_bytes!`) to a properly aligned `&[u32]`.
/// `include_bytes!` returns `&[u8]` (alignment 1), but Vulkan requires `&[u32]`
/// (alignment 4). We copy through an aligned `Vec<u32>` to avoid bytemuck panics.
fn spirv_to_u32(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

fn create_ui_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_ui_descriptor_set_layout: {e:?}"))
}

fn allocate_ui_descriptor_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet> {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&alloc_info) }
        .map_err(|e| anyhow!("allocate_ui_descriptor_set: {e:?}"))?;
    Ok(sets[0])
}

fn update_ui_descriptor_set(
    device: &ash::Device,
    set: vk::DescriptorSet,
    block_view: vk::ImageView,
    block_sampler: vk::Sampler,
    font_view: vk::ImageView,
    font_sampler: vk::Sampler,
) {
    let block_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(block_view)
        .sampler(block_sampler);
    let font_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(font_view)
        .sampler(font_sampler);
    let block_infos = [block_info];
    let font_infos = [font_info];
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&block_infos),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&font_infos),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

fn create_ui_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(8); // vec2 screen_size
    let set_layouts = [set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_ui_pipeline_layout: {e:?}"))
}

fn create_ui_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/ui.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/ui.frag.spv"));
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("ui vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("ui frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::ui::UiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R8G8B8A8_UNORM)
            .offset(16),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // No depth test/write for UI — it draws on top of everything.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);

    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_ui_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── Sky pipeline helpers ─────────────────────────────────────────────────

fn create_sky_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_sky_descriptor_set_layout: {e:?}"))
}

fn create_sky_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80); // mat4 inverse view-proj + vec4 camera_pos
    let set_layouts = [set_layout];
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_sky_pipeline_layout: {e:?}"))
}

fn create_sky_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/sky.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/sky.frag.spv"));
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("sky vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("sky frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    // No vertex input — the sky shader generates a full-screen triangle from
    // gl_VertexIndex.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    // Sky pass: depth test LESS_OR_EQUAL (so it draws at depth=1, behind everything),
    // no depth write (so chunks can overwrite it).
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

    // No blending for the sky — it's the background.
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA);

    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_sky_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

// ── Shadow + Post pipeline helpers ─────────────────────────────────────────

fn create_shadow_render_pass(device: &ash::Device, depth_format: vk::Format) -> Result<vk::RenderPass> {
    let depth_attachment = vk::AttachmentDescription::default()
        .format(depth_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let depth_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref);

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        )
        .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

    let attachments = [depth_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_shadow_render_pass: {e:?}"))
}

fn create_shadow_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0)
        .size(80);
    let push_ranges = [push_range];
    let create_info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_shadow_pipeline_layout: {e:?}"))
}

fn create_shadow_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/shadow.frag.spv"));
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("shadow vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("shadow frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(std::mem::size_of::<crate::Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attrs = [
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(12),
        vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(2)
            .format(vk::Format::R32_SFLOAT)
            .offset(20),
    ];
    let vertex_bindings = [vertex_binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_bindings)
        .vertex_attribute_descriptions(&vertex_attrs);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0)
        .depth_bias_enable(true)
        .depth_bias_constant_factor(2.0)
        .depth_bias_slope_factor(1.5);

    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let color_blend = vk::PipelineColorBlendStateCreateInfo::default().logic_op_enable(false);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_shadow_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

fn create_post_render_pass(device: &ash::Device, color_format: vk::Format) -> Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::default()
        .format(color_format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_ref = vk::AttachmentReference::default()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_refs = [color_ref];
    let subpass = vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_refs);

    let dependency = vk::SubpassDependency::default()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = [color_attachment];
    let subpasses = [subpass];
    let dependencies = [dependency];
    let create_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);

    unsafe { device.create_render_pass(&create_info, None) }
        .map_err(|e| anyhow!("create_post_render_pass: {e:?}"))
}

fn create_post_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout> {
    let push_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(16);
    let push_ranges = [push_range];
    let set_layouts = [descriptor_set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_ranges);
    unsafe { device.create_pipeline_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_post_pipeline_layout: {e:?}"))
}

fn create_post_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
) -> Result<vk::Pipeline> {
    let vert_spv = include_bytes!(concat!(env!("OUT_DIR"), "/post.vert.spv"));
    let frag_spv = include_bytes!(concat!(env!("OUT_DIR"), "/post.frag.spv"));
    let vert_code = spirv_to_u32(vert_spv);
    let frag_code = spirv_to_u32(frag_spv);
    let vert_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&vert_code),
            None,
        )
    }
    .map_err(|e| anyhow!("post vert shader: {e:?}"))?;
    let frag_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&frag_code),
            None,
        )
    }
    .map_err(|e| anyhow!("post frag shader: {e:?}"))?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_module)
            .name(c"main"),
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false);
    let attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let blend_attachments = [attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(&blend_attachments)
        .logic_op_enable(false);
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

    let create_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterizer)
        .multisample_state(&multisampling)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);

    let result = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    unsafe {
        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
    }
    let pipelines = result.map_err(|(_p, e)| anyhow!("create_post_pipeline: {e:?}"))?;
    Ok(pipelines.into_iter().next().unwrap())
}

fn create_post_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&create_info, None) }
        .map_err(|e| anyhow!("create_post_descriptor_set_layout: {e:?}"))
}

fn create_shadow_layer_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    base_array_layer: u32,
) -> Result<vk::ImageView> {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer,
            layer_count: 1,
        });
    unsafe { device.create_image_view(&create_info, None) }
        .map_err(|e| anyhow!("create_shadow_layer_view: {e:?}"))
}

fn create_framebuffer_with(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    attachments: &[vk::ImageView],
    extent: vk::Extent2D,
) -> Result<vk::Framebuffer> {
    let create_info = vk::FramebufferCreateInfo::default()
        .render_pass(render_pass)
        .attachments(attachments)
        .width(extent.width)
        .height(extent.height)
        .layers(1);
    unsafe { device.create_framebuffer(&create_info, None) }
        .map_err(|e| anyhow!("create_framebuffer_with: {e:?}"))
}


