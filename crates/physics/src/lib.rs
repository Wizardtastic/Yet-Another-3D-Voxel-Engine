//! `voxel-physics` — voxel-aware collision and spatial queries.
//!
//! - `swept_aabb` — move an AABB through the voxel grid, stopping at each axis
//!   collision (the standard "resolve per axis" voxel collision used by
//!   Minecraft-like controllers).
//! - `raycast_voxels` — DDA voxel traversal returning the first solid hit and
//!   the face normal, used for block targeting (breaking/placing).

use voxel_core::math::ChunkPos;
use voxel_core::{
    math::{world_to_block, Aabb, Ray},
    BlockId,
};
use voxel_world::chunk::Chunk;
use voxel_world::World;

/// A single voxel ray hit.
#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    /// Block coordinate that was hit.
    pub block: glam::IVec3,
    /// Face normal of the hit (points back toward the ray origin).
    pub normal: glam::IVec3,
    /// Distance along the ray to the hit.
    pub dist: f32,
    /// Block id at the hit (sampled at hit time).
    pub block_id: BlockId,
}

/// DDA voxel raycast. Walks voxel cells along `ray` and returns the first block
/// whose id is not air (and is solid per the optional `solid` filter). Returns
/// `None` if nothing is hit within `ray.max_dist`.
pub fn raycast_voxels(world: &World, ray: Ray) -> Option<RayHit> {
    use glam::IVec3;

    let origin = ray.origin;
    let dir = ray.dir;
    if dir.length_squared() < 1e-12 {
        return None;
    }

    let mut block = world_to_block(origin);
    let step = IVec3::new(
        if dir.x > 0.0 { 1 } else { -1 },
        if dir.y > 0.0 { 1 } else { -1 },
        if dir.z > 0.0 { 1 } else { -1 },
    );

    let inv_x = if dir.x.abs() > 1e-12 {
        1.0 / dir.x.abs()
    } else {
        f32::INFINITY
    };
    let inv_y = if dir.y.abs() > 1e-12 {
        1.0 / dir.y.abs()
    } else {
        f32::INFINITY
    };
    let inv_z = if dir.z.abs() > 1e-12 {
        1.0 / dir.z.abs()
    } else {
        f32::INFINITY
    };

    // Distance to the first voxel boundary along each axis (f32 deltas).
    let mut t_max = glam::Vec3::new(
        boundary_dist(origin.x, step.x) * inv_x,
        boundary_dist(origin.y, step.y) * inv_y,
        boundary_dist(origin.z, step.z) * inv_z,
    );
    let t_delta = glam::Vec3::new(inv_x, inv_y, inv_z);

    let mut normal = IVec3::ZERO;
    let mut t = 0.0f32;
    let abs_x = dir.x.abs();
    let abs_y = dir.y.abs();
    let abs_z = dir.z.abs();

    // Acquire chunks read lock once for the entire raycast.
    let chunks_guard = world.chunks_ref().read();

    while t <= ray.max_dist {
        let id = World::get_block_guarded(&chunks_guard, block.x, block.y, block.z);
        if !id.is_air() {
            // Skip water/liquid blocks so the player can hit blocks through
            // water. Fall through to the DDA advancement to step past.
            if world.registry_ref().is_liquid(id) {
                // Fall through to advance past this water block.
            } else {
                return Some(RayHit {
                    block,
                    normal,
                    dist: t,
                    block_id: id,
                });
            }
        }

        // Advance to the nearest voxel boundary.
        // When axes are tied, prefer the one with the largest direction component
        // (most perpendicular to the face), giving correct face normals at edges.
        if t_max.x < t_max.y || (t_max.x == t_max.y && abs_x > abs_y) {
            if t_max.x < t_max.z || (t_max.x == t_max.z && abs_x > abs_z) {
                t = t_max.x;
                t_max.x += t_delta.x;
                block.x += step.x;
                normal = IVec3::new(-step.x, 0, 0);
            } else {
                t = t_max.z;
                t_max.z += t_delta.z;
                block.z += step.z;
                normal = IVec3::new(0, 0, -step.z);
            }
        } else if t_max.y < t_max.z || (t_max.y == t_max.z && abs_y > abs_z) {
            t = t_max.y;
            t_max.y += t_delta.y;
            block.y += step.y;
            normal = IVec3::new(0, -step.y, 0);
        } else {
            t = t_max.z;
            t_max.z += t_delta.z;
            block.z += step.z;
            normal = IVec3::new(0, 0, -step.z);
        }
    }
    None
}

/// Distance from `p` to the next integer boundary in `step` direction.
#[inline]
fn boundary_dist(p: f32, step: i32) -> f32 {
    if step > 0 {
        (p.floor() + 1.0) - p
    } else if step < 0 {
        // For negative step: distance to the next lower integer boundary.
        // If p is exactly on an integer boundary (e.g. p=0, step=-1), we
        // must go one full block back to reach the next voxel.
        let frac = p - p.floor();
        if frac == 0.0 {
            1.0
        } else {
            frac
        }
    } else {
        f32::INFINITY
    }
}

/// Result of a swept collision move: where the box ended up and whether each
/// axis was blocked.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveResult {
    pub new_pos: glam::Vec3,
    /// True if the box collided on +X / -X / +Y / -Y / +Z / -Z respectively.
    pub hit: [bool; 6],
    pub on_ground: bool,
}

/// Move `box_half`-sized AABB centred at `pos` by `delta`, sliding along voxel
/// surfaces. Resolves X, then Y, then Z independently (the classic approach).
/// `pos` is the box *centre*; `box_half` is the half-extent.
pub fn swept_aabb(
    world: &World,
    pos: glam::Vec3,
    box_half: glam::Vec3,
    delta: glam::Vec3,
) -> MoveResult {
    let mut p = pos;
    let mut hit = [false; 6];
    let chunks_guard = world.chunks_ref().read();

    // Y axis first so that AABB height changes (crouch↔stand) resolve
    // vertically before horizontal sweeps, preventing the taller AABB
    // from intersecting with blocks below the ground surface.
    p = move_axis(
        &chunks_guard,
        world.registry_ref(),
        p,
        box_half,
        glam::Vec3::new(0.0, delta.y, 0.0),
        &mut hit,
        1,
    );
    // X axis
    p = move_axis(
        &chunks_guard,
        world.registry_ref(),
        p,
        box_half,
        glam::Vec3::new(delta.x, 0.0, 0.0),
        &mut hit,
        0,
    );
    // Z axis
    p = move_axis(
        &chunks_guard,
        world.registry_ref(),
        p,
        box_half,
        glam::Vec3::new(0.0, 0.0, delta.z),
        &mut hit,
        2,
    );

    let on_ground = hit[3] && delta.y < 0.0; // hit -Y while moving down
    MoveResult {
        new_pos: p,
        hit,
        on_ground,
    }
}

fn move_axis(
    chunks: &std::collections::HashMap<ChunkPos, Chunk>,
    reg: &voxel_world::registry::BlockRegistry,
    pos: glam::Vec3,
    half: glam::Vec3,
    delta: glam::Vec3,
    hit: &mut [bool; 6],
    axis: usize,
) -> glam::Vec3 {
    let mut new_pos = pos + delta;
    let aabb = Aabb::from_center_size(new_pos, half * 2.0);

    // Voxels overlapping the box on this axis.
    let min_b = world_to_block(aabb.min);
    let max_b = world_to_block(aabb.max - glam::Vec3::splat(0.001));

    for by in min_b.y..=max_b.y {
        for bz in min_b.z..=max_b.z {
            for bx in min_b.x..=max_b.x {
                if !World::is_solid_guarded(chunks, reg, bx, by, bz) {
                    continue;
                }
                // Collision on this axis: push the box back to the block face.
                match axis {
                    0 => {
                        if delta.x > 0.0 {
                            new_pos.x = bx as f32 - half.x - 1e-3;
                            hit[1] = true; // +X blocked
                        } else if delta.x < 0.0 {
                            new_pos.x = (bx + 1) as f32 + half.x + 1e-3;
                            hit[0] = true; // -X blocked
                        }
                    }
                    1 => {
                        if delta.y > 0.0 {
                            new_pos.y = by as f32 - half.y - 1e-3;
                            hit[2] = true; // +Y blocked (hit ceiling)
                        } else if delta.y < 0.0 {
                            new_pos.y = (by + 1) as f32 + half.y + 1e-3;
                            hit[3] = true; // -Y blocked (on ground)
                        }
                    }
                    2 => {
                        if delta.z > 0.0 {
                            new_pos.z = bz as f32 - half.z - 1e-3;
                            hit[5] = true; // +Z blocked
                        } else if delta.z < 0.0 {
                            new_pos.z = (bz + 1) as f32 + half.z + 1e-3;
                            hit[4] = true; // -Z blocked
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    new_pos
}

/// Check whether an AABB centred at `pos` with half-extent `half` intersects any
/// solid voxel. Used for spawn/ground checks.
pub fn intersects_solid(world: &World, pos: glam::Vec3, half: glam::Vec3) -> bool {
    let aabb = Aabb::from_center_size(pos, half * 2.0);
    let min_b = world_to_block(aabb.min);
    let max_b = world_to_block(aabb.max - glam::Vec3::splat(0.001));
    let chunks_guard = world.chunks_ref().read();
    for by in min_b.y..=max_b.y {
        for bz in min_b.z..=max_b.z {
            for bx in min_b.x..=max_b.x {
                if World::is_solid_guarded(&chunks_guard, world.registry_ref(), bx, by, bz) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::math::world_to_block;
    use voxel_core::Ray;
    use voxel_world::world::World;

    #[test]
    fn raycast_voxels_air() {
        let world = World::new(42);
        let ray = Ray::new(glam::Vec3::new(0.5, 100.0, 0.5), glam::Vec3::new(0.0, -1.0, 0.0), 10.0);
        assert!(raycast_voxels(&world, ray).is_none());
    }

    #[test]
    fn boundary_dist_positive_step() {
        assert!((boundary_dist(0.5, 1) - 0.5).abs() < f32::EPSILON);
        assert!((boundary_dist(0.0, 1) - 1.0).abs() < f32::EPSILON);
        assert!((boundary_dist(0.99, 1) - 0.01).abs() < 1e-5);
    }

    #[test]
    fn boundary_dist_negative_step() {
        assert!((boundary_dist(-0.5, -1) - 0.5).abs() < f32::EPSILON);
        assert!((boundary_dist(0.0, -1) - 1.0).abs() < f32::EPSILON);
        assert!((boundary_dist(-0.01, -1) - 0.99).abs() < 1e-5);
    }

    #[test]
    fn raycast_hits_block_with_neg_dir() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(5, 5, 5, voxel_core::BlockId(2));
        world.insert_chunk(cp, chunk);
        let ray = Ray::new(
            glam::Vec3::new(5.5, 10.0, 5.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            20.0,
        );
        let hit = raycast_voxels(&world, ray).expect("should hit");
        assert_eq!(hit.block.y, 5);
    }

    #[test]
    fn raycast_returns_face_normal() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(5, 5, 5, voxel_core::BlockId(2));
        world.insert_chunk(cp, chunk);
        // Looking down at top face.
        let ray = Ray::new(
            glam::Vec3::new(5.5, 10.0, 5.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            20.0,
        );
        let hit = raycast_voxels(&world, ray).expect("should hit");
        assert_eq!(hit.normal, glam::IVec3::new(0, 1, 0));
    }

    #[test]
    fn raycast_zero_direction() {
        let world = World::new(42);
        let ray = Ray::new(
            glam::Vec3::new(0.5, 0.5, 0.5),
            glam::Vec3::ZERO,
            10.0,
        );
        assert!(raycast_voxels(&world, ray).is_none());
    }

    #[test]
    fn swept_aabb_no_movement() {
        let world = World::new(42);
        let pos = glam::Vec3::new(0.5, 50.0, 0.5);
        let res = swept_aabb(&world, pos, glam::Vec3::splat(0.3), glam::Vec3::ZERO);
        assert_eq!(res.new_pos, pos);
        assert!(!res.on_ground);
    }

    #[test]
    fn swept_aabb_wall_collision() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        let stone = world.registry().id_of("stone").unwrap();
        // Place a stone block at x=4 (right where the player is about to go).
        for y in 7..10 {
            chunk.set(4, y, 5, stone);
        }
        world.insert_chunk(cp, chunk);
        // Player starts inside the stone (overlapping). Moving +X should
        // resolve by pushing the player back to the block face.
        let pos = glam::Vec3::new(4.4, 8.0, 5.5);
        let res = swept_aabb(
            &world,
            pos,
            glam::Vec3::new(0.3, 0.9, 0.3),
            glam::Vec3::new(0.1, 0.0, 0.0),
        );
        assert!(res.hit[1], "+X should be blocked, hit={:?}", res.hit);
    }

    #[test]
    fn intersects_solid_empty() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let chunk = Chunk::new(cp);
        world.insert_chunk(cp, chunk);
        assert!(!intersects_solid(
            &world,
            glam::Vec3::new(8.0, 8.0, 8.0),
            glam::Vec3::splat(0.3)
        ));
    }

    #[test]
    fn intersects_solid_finds_block() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(5, 5, 5, voxel_core::BlockId(2));
        world.insert_chunk(cp, chunk);
        assert!(intersects_solid(
            &world,
            glam::Vec3::new(5.5, 5.5, 5.5),
            glam::Vec3::splat(0.3)
        ));
    }
}
