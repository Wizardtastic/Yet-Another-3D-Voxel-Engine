//! Block interaction: breaking and placing using the player's look ray.
//!
//! The engine calls `BlockAction::apply` each frame with the current clicks.
//! Breaking replaces the targeted block with air; placing puts the held block
//! adjacent to the targeted face (classic Minecraft behaviour). Both request a
//! remesh of the affected chunk (and neighbours) via the `ChunkStreamer`.

use std::sync::Arc;

use voxel_core::{math::block_to_chunk, BlockId};
use voxel_physics::{raycast_voxels, RayHit};
use voxel_world::{ChunkStreamer, World};

use crate::input::Clicks;
use crate::inv::Hotbar;
use crate::undo::BlockEdit;

/// How far the player can reach to interact with a block (Minecraft creative is
/// ~5; survival ~4.5). We use 5.0.
pub const REACH: f32 = 5.0;

/// The result of attempting a break or place this frame.
#[derive(Clone, Debug, Default)]
pub struct ActionResult {
    pub broke: bool,
    pub placed: bool,
    pub target: Option<RayHit>,
    /// Block edits to record for undo.
    pub edits: Vec<BlockEdit>,
}

pub struct BlockAction;

impl BlockAction {
    /// Apply this frame's clicks against the world. `eye` is the camera/eye
    /// position; `look_dir` the view direction. Returns what happened.
    pub fn apply(
        world: &Arc<World>,
        streamer: &ChunkStreamer,
        hotbar: &mut Hotbar,
        eye: glam::Vec3,
        look_dir: glam::Vec3,
        clicks: Clicks,
        player_pos: glam::Vec3,
    ) -> ActionResult {
        let ray = voxel_core::Ray::new(eye, look_dir, REACH);
        let hit = raycast_voxels(world, ray);
        let mut out = ActionResult {
            target: hit,
            ..Default::default()
        };

        if let Some(h) = hit {
            if clicks.left {
                let old = world.get_block(h.block.x, h.block.y, h.block.z);
                let reg = world.registry();
                if !reg.get(old).breakable {
                    // Unbreakable block (e.g. bedrock). Skip silently.
                } else if world.set_block(h.block.x, h.block.y, h.block.z, BlockId::AIR) {
                    out.broke = true;
                    out.edits.push(BlockEdit {
                        x: h.block.x,
                        y: h.block.y,
                        z: h.block.z,
                        old_block: old.0,
                        new_block: BlockId::AIR.0,
                    });
                    let cp = block_to_chunk(h.block);
                    streamer.request_remesh(cp);
                    for n in neighbours(cp) {
                        streamer.request_remesh(n);
                    }
                }                } else if clicks.right {
                    let id = hotbar.selected_block();
                    let reg = world.registry();
                    let name = reg.get(id).name.clone();

                    if name.as_ref() == "bucket" || name.as_ref() == "water_bucket" {
                        if name.as_ref() == "bucket" && world.is_water_source(h.block.x, h.block.y, h.block.z) {
                        // Empty bucket on water: pick up source (single block).
                        let bx = h.block.x;
                        let by = h.block.y;
                        let bz = h.block.z;
                        if world.remove_water(bx, by, bz) {
                            out.placed = true;
                            // Swap to water_bucket.
                            if let Some(wb_id) = reg.id_of("water_bucket") {
                                hotbar.set_slot(hotbar.selected, wb_id);
                            }
                            let cp = block_to_chunk(h.block);
                            streamer.request_remesh(cp);
                            for n in neighbours(cp) {
                                streamer.request_remesh(n);
                            }
                        }
                    } else if name.as_ref() == "water_bucket" {
                        // Water bucket: place in the targeted block if replaceable,
                        // otherwise adjacent to the hit face.
                        let target = world.get_block(h.block.x, h.block.y, h.block.z);
                        let place = if target.is_air()
                            || (!reg.is_solid(target) && !reg.is_liquid(target))
                        {
                            h.block
                        } else {
                            h.block + h.normal
                        };
                        let place_block = world.get_block(place.x, place.y, place.z);
                        if reg.get(place_block).replaceable
                            && world.place_water(place.x, place.y, place.z)
                        {
                            out.placed = true;
                            // Swap to empty bucket.
                            if let Some(b_id) = reg.id_of("bucket") {
                                hotbar.set_slot(hotbar.selected, b_id);
                            }
                            let cp = block_to_chunk(place);
                            streamer.request_remesh(cp);
                            for n in neighbours(cp) {
                                streamer.request_remesh(n);
                            }
                        }
                    }
                } else if !id.is_air() {
                    // Normal block placement.
                    let place = h.block + h.normal;
                    let player_aabb_min = player_pos - glam::Vec3::new(0.3, 0.9, 0.3);
                    let player_aabb_max = player_pos + glam::Vec3::new(0.3, 0.9, 0.3);
                    let block_min = glam::Vec3::new(place.x as f32, place.y as f32, place.z as f32);
                    let block_max = block_min + glam::Vec3::ONE;
                    let overlaps = player_aabb_min.x < block_max.x
                        && player_aabb_max.x > block_min.x
                        && player_aabb_min.y < block_max.y
                        && player_aabb_max.y > block_min.y
                        && player_aabb_min.z < block_max.z
                        && player_aabb_max.z > block_min.z;
                    if overlaps {
                        return out;
                    }
                    let old = world.get_block(place.x, place.y, place.z);
                    if reg.get(old).replaceable
                        && world.set_block(place.x, place.y, place.z, id)
                    {
                        out.placed = true;
                        out.edits.push(BlockEdit {
                            x: place.x,
                            y: place.y,
                            z: place.z,
                            old_block: old.0,
                            new_block: id.0,
                        });
                        let cp = block_to_chunk(place);
                        streamer.request_remesh(cp);
                        for n in neighbours(cp) {
                            streamer.request_remesh(n);
                        }
                    }
                }
            }
        }

        out
    }
}

fn neighbours(p: voxel_core::ChunkPos) -> [voxel_core::ChunkPos; 6] {
    [
        voxel_core::ChunkPos::new(p.x() - 1, p.y(), p.z()),
        voxel_core::ChunkPos::new(p.x() + 1, p.y(), p.z()),
        voxel_core::ChunkPos::new(p.x(), p.y() - 1, p.z()),
        voxel_core::ChunkPos::new(p.x(), p.y() + 1, p.z()),
        voxel_core::ChunkPos::new(p.x(), p.y(), p.z() - 1),
        voxel_core::ChunkPos::new(p.x(), p.y(), p.z() + 1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::Clicks;
    use crate::inv::Hotbar;
    use voxel_core::ChunkPos;
    use voxel_world::chunk::Chunk;

    fn world_with_block(block: BlockId) -> std::sync::Arc<World> {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(5, 5, 5, block);
        world.insert_chunk(cp, chunk);
        world
    }

    #[test]
    fn break_replaces_block_with_air() {
        let world = world_with_block(BlockId(2));
        let streamer = ChunkStreamer::spawn(world.clone(), Default::default()).unwrap();
        let mut hotbar = Hotbar::new();
        let result = BlockAction::apply(
            &world,
            &streamer,
            &mut hotbar,
            glam::Vec3::new(5.5, 10.0, 5.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            Clicks { left: true, right: false },
            glam::Vec3::new(5.5, 50.0, 5.5),
        );
        assert!(result.broke);
        assert_eq!(world.get_block(5, 5, 5), BlockId::AIR);
        assert!(!result.edits.is_empty());
    }

    #[test]
    fn break_bedrock_is_noop() {
        let bedrock_id = world_with_block(BlockId(2))
            .registry()
            .id_of("bedrock")
            .unwrap();
        let world = world_with_block(bedrock_id);
        let streamer = ChunkStreamer::spawn(world.clone(), Default::default()).unwrap();
        let mut hotbar = Hotbar::new();
        let result = BlockAction::apply(
            &world,
            &streamer,
            &mut hotbar,
            glam::Vec3::new(5.5, 10.0, 5.5),
            glam::Vec3::new(0.0, -1.0, 0.0),
            Clicks { left: true, right: false },
            glam::Vec3::new(5.5, 50.0, 5.5),
        );
        assert!(!result.broke);
        assert_eq!(world.get_block(5, 5, 5), bedrock_id);
    }

    #[test]
    fn place_adjacent_to_block_succeeds() {
        let stone_id = world_with_block(BlockId(2))
            .registry()
            .id_of("stone")
            .unwrap();
        let world = world_with_block(stone_id);
        let streamer = ChunkStreamer::spawn(world.clone(), Default::default()).unwrap();
        let mut hotbar = Hotbar::new();
        hotbar.set_slot(0, stone_id);
        // Look at the stone from the side (positive X direction). The place
        // position would be (4, 5, 5) which is air — placing should succeed.
        let result = BlockAction::apply(
            &world,
            &streamer,
            &mut hotbar,
            glam::Vec3::new(0.5, 5.5, 5.5),
            glam::Vec3::new(1.0, 0.0, 0.0),
            Clicks { left: false, right: true },
            glam::Vec3::new(0.5, 50.0, 5.5),
        );
        assert!(result.placed);
        // The placed block is at hit.block + hit.normal = (5,5,5) + (-1,0,0) = (4,5,5).
        assert_eq!(world.get_block(4, 5, 5), stone_id);
    }
}
