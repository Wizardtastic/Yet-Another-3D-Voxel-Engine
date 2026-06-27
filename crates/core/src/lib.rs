//! `voxel-core` — shared engine primitives with no heavy dependencies.
//!
//! See the crate-level docs in the source tree. This module re-exports the
//! public submodules so consumers can `use voxel_core::*`.

pub mod block;
pub mod camera;
pub mod constants;
pub mod math;

pub use block::BlockId;
pub use camera::{Camera, Frustum};
pub use constants::*;
pub use math::{Aabb, BlockPos, ChunkPos, Ray};

/// Engine version string, baked at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Semantic placeholder so the crate links cleanly during scaffolding.
pub fn version() -> &'static str {
    VERSION
}
