//! Engine-wide numeric constants. Centralised so subsystems never disagree
//! about world geometry.

/// Side length of a single chunk (a cubic chunk section), in blocks.
pub const CHUNK_SIZE: i32 = 16;
/// [`CHUNK_SIZE`] as a `usize` for array indexing.
pub const CHUNK_SIZE_US: usize = 16;
/// Number of blocks in a cubic chunk: 16³ = 4096.
pub const CHUNK_CUBED: usize = CHUNK_SIZE_US * CHUNK_SIZE_US * CHUNK_SIZE_US;

/// World height measured in stacked chunks. 16 chunks → 256 blocks (Minecraft-like).
pub const WORLD_HEIGHT_CHUNKS: i32 = 16;
/// World height in blocks.
pub const WORLD_HEIGHT_BLOCKS: i32 = CHUNK_SIZE * WORLD_HEIGHT_CHUNKS;

/// Vertical index of the lowest chunk (chunks below this are not generated).
pub const MIN_CHUNK_Y: i32 = 0;
/// Vertical index of the highest chunk (exclusive).
pub const MAX_CHUNK_Y: i32 = WORLD_HEIGHT_CHUNKS;

/// Sea level in block coordinates. Terrain below this that is air becomes water.
pub const SEA_LEVEL: i32 = 62;

/// Atlas tile resolution in pixels (textures are square).
pub const ATLAS_TILE_SIZE: u32 = 16;
/// Atlas grid dimension in tiles (16×16 = 256 tiles).
pub const ATLAS_TILES: u32 = 16;
