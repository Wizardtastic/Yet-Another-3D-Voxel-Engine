//! `voxel-world` — procedural voxel world: storage, generation, meshing, streaming.
//!
//! See crate root docs in source. Submodules:
//! - `registry` — block definitions and their render/physics properties
//! - `chunk`    — `Chunk` storage (16³) with palette + get/set + neighbours
//! - `gen`      — layered-noise terrain, biomes, caves, ores, trees, rivers
//! - `mesh`     — face-culled chunk meshing producing GPU vertex data
//! - `stream`   — background chunk load/unload around a focus point
//! - `world`    — `World` facade tying storage + streaming + queries together

pub mod chunk;
pub mod gen;
pub mod light;
pub mod mesh;
pub mod registry;
pub mod save;
pub mod stream;
pub mod water;
pub mod world;

pub use chunk::Chunk;
pub use gen::{BiomeId, TerrainGenerator};
pub use mesh::{ChunkMesh, ChunkMeshBundle, ChunkMesher};
pub use registry::{BlockKind, BlockRegistry};
pub use stream::{ChunkStreamEvent, ChunkStreamer, StreamConfig};
pub use world::World;
