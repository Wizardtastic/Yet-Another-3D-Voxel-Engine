//! Small math layer over `glam` plus voxel coordinate helpers.
//!
//! Coordinate conventions:
//! - *World space* is right-handed metres; +Y up. One block == one metre.
//! - *Block space* is integer block coordinates in `[0, WORLD_HEIGHT_BLOCKS)`.
//! - *Chunk space* is integer chunk coordinates; chunk `(cx, cy, cz)` owns blocks
//!   `[cx*16, cx*16+16)` etc.

use glam::{IVec3, Vec3};
use std::ops::{Add, Sub};

use crate::constants::{CHUNK_SIZE, CHUNK_SIZE_US};

/// Re-export the glam types the rest of the engine uses, so crates don't each
/// depend on a specific glam version for these primitives.
pub use glam::{Mat4, Quat, UVec3, Vec2, Vec3 as Vec3f, Vec4};

/// Integer chunk coordinate. Used as a `HashMap` key, so it is `Hash` + `Eq`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos(pub IVec3);

impl ChunkPos {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }
    #[inline]
    pub fn x(self) -> i32 {
        self.0.x
    }
    #[inline]
    pub fn y(self) -> i32 {
        self.0.y
    }
    #[inline]
    pub fn z(self) -> i32 {
        self.0.z
    }
}

impl Add<IVec3> for ChunkPos {
    type Output = ChunkPos;
    #[inline]
    fn add(self, rhs: IVec3) -> ChunkPos {
        ChunkPos(self.0 + rhs)
    }
}
impl Sub for ChunkPos {
    type Output = IVec3;
    #[inline]
    fn sub(self, rhs: ChunkPos) -> IVec3 {
        self.0 - rhs.0
    }
}

/// Integer block coordinate in world space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct BlockPos(pub IVec3);

impl BlockPos {
    #[inline]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }
}

/// Axis-aligned bounding box in world space.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn from_center_size(center: Vec3, size: Vec3) -> Self {
        let half = size * 0.5;
        Self::new(center - half, center + half)
    }

    #[inline]
    pub fn center(self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    #[inline]
    pub fn size(self) -> Vec3 {
        self.max - self.min
    }

    #[inline]
    pub fn contains_point(self, p: Vec3) -> bool {
        p.x >= self.min.x
            && p.x <= self.max.x
            && p.y >= self.min.y
            && p.y <= self.max.y
            && p.z >= self.min.z
            && p.z <= self.max.z
    }
}

/// A ray with a normalised direction and a maximum reach.
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub dir: Vec3,
    pub max_dist: f32,
}

impl Ray {
    #[inline]
    pub fn new(origin: Vec3, dir: Vec3, max_dist: f32) -> Self {
        Self {
            origin,
            dir: dir.normalize_or_zero(),
            max_dist,
        }
    }
    #[inline]
    pub fn at(self, t: f32) -> Vec3 {
        self.origin + self.dir * t
    }
}

/// Convert a world-space position to its containing block coordinate.
/// Fast floor using truncation (avoids hardware `floor()` instruction).
#[inline]
pub fn world_to_block(p: Vec3) -> IVec3 {
    // Truncation rounds toward zero. For floor we need toward -inf:
    // subtract 1 for negative values whose fractional part is nonzero.
    IVec3::new(fast_floor(p.x), fast_floor(p.y), fast_floor(p.z))
}

#[inline]
fn fast_floor(v: f32) -> i32 {
    let i = v as i32;
    if v < 0.0 && v != i as f32 {
        i - 1
    } else {
        i
    }
}

/// Convert a block coordinate to its containing chunk coordinate.
/// Uses arithmetic shift for power-of-two CHUNK_SIZE (16).
#[inline]
pub fn block_to_chunk(b: IVec3) -> ChunkPos {
    ChunkPos(IVec3::new(
        b.x.div_euclid(CHUNK_SIZE),
        b.y.div_euclid(CHUNK_SIZE),
        b.z.div_euclid(CHUNK_SIZE),
    ))
}

/// The world-space block origin (minimum corner) of a chunk.
#[inline]
pub fn chunk_origin(c: ChunkPos) -> IVec3 {
    c.0 * CHUNK_SIZE
}

/// Linear array index for a block local coordinate in `[0, CHUNK_SIZE)`.
/// Layout is x-fastest: `idx = (y*16 + z)*16 + x`.
#[inline]
pub fn local_index(x: i32, y: i32, z: i32) -> usize {
    (y as usize * CHUNK_SIZE_US + z as usize) * CHUNK_SIZE_US + x as usize
}

/// Inverse of [`local_index`]: returns `(x, y, z)` for a flat index.
#[inline]
pub fn index_to_local(idx: usize) -> (i32, i32, i32) {
    let x = (idx % CHUNK_SIZE_US) as i32;
    let yz = idx / CHUNK_SIZE_US;
    let z = (yz % CHUNK_SIZE_US) as i32;
    let y = (yz / CHUNK_SIZE_US) as i32;
    (x, y, z)
}

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::constants::CHUNK_CUBED;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn block_to_chunk_roundtrip(x in -1000i32..1000, y in 0i32..256, z in -1000i32..1000) {
            let block = IVec3::new(x, y, z);
            let cp = block_to_chunk(block);
            let origin = chunk_origin(cp);
            // block is within the chunk's 16x16x16 volume
            prop_assert!(x >= origin.x && x < origin.x + CHUNK_SIZE);
            prop_assert!(y >= origin.y && y < origin.y + CHUNK_SIZE);
            prop_assert!(z >= origin.z && z < origin.z + CHUNK_SIZE);
        }

        #[test]
        fn local_index_roundtrip(x in 0i32..16, y in 0i32..16, z in 0i32..16) {
            let idx = local_index(x, y, z);
            let (rx, ry, rz) = index_to_local(idx);
            prop_assert_eq!((x, y, z), (rx, ry, rz));
        }

        #[test]
        fn local_index_in_range(x in 0i32..16, y in 0i32..16, z in 0i32..16) {
            let idx = local_index(x, y, z);
            prop_assert!(idx < CHUNK_CUBED, "index {idx} out of range");
        }

        #[test]
        fn chunk_pos_add_sub(cp in (0i32..10, 0i32..10, 0i32..10).prop_map(|(x,y,z)| ChunkPos::new(x,y,z)),
                             offset in (-3i32..3, -3i32..3, -3i32..3).prop_map(|(x,y,z)| IVec3::new(x,y,z))) {
            let cp2 = cp + offset;
            prop_assert_eq!(cp2 - cp, offset);
        }

        #[test]
        fn aabb_center_in_bounds(min_x in -100f32..0.0, size in 0.1f32..10.0) {
            let min = Vec3::new(min_x, 0.0, 0.0);
            let size_v = Vec3::splat(size);
            let aabb = Aabb::new(min, min + size_v);
            prop_assert!(aabb.contains_point(aabb.center()));
        }
    }
}
