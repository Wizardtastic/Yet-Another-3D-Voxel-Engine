//! `voxel-render` — Vulkan rendering backend.
//!
//! See crate root docs. Public surface:
//! - [`Renderer`] — the high-level facade owned by the engine.
//! - [`atlas`] — name-based texture atlas loaded from PNG files.
//! - [`Vertex`] — the GPU vertex layout (24 bytes), matching the chunk mesher.
//!
//! The renderer is intentionally decoupled from `voxel-world`: it accepts chunk
//! meshes as raw bytes (`&[u8]` vertices + `&[u32]` indices) plus a chunk world
//! origin, so worldgen/meshing can evolve without touching the renderer.

pub mod alloc;
pub mod atlas;
pub mod buffer;
pub mod renderer;
pub mod texture;
pub mod ui;

pub use atlas::{build_atlas_with_textures, Atlas};
pub use renderer::{ChunkUpload, GpuTimings, MeshPass, Renderer, RendererConfig};
pub use texture::AtlasTexture;
pub use ui::{FontAtlas, UiDrawData, UiVertex};

use bytemuck::{Pod, Zeroable};

/// GPU vertex layout (24 bytes), matching `voxel_world::mesh::ChunkVertex`.
/// Kept here as the rendering-side contract; the mesher produces the same layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
    pub light: f32,
}
