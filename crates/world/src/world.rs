//! `World` — the shared facade over chunk storage, block registry, terrain
//! generation, and chunk meshes. Safe to share across threads via `Arc`; all
//! mutation goes through internal `RwLock`s.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use parking_lot::RwLock;

use glam::IVec3;
use voxel_core::{
    math::{block_to_chunk, chunk_origin, ChunkPos},
    BlockId, BlockPos, WORLD_HEIGHT_BLOCKS,
};

use crate::{chunk::Chunk, gen::TerrainGenerator, mesh::ChunkMeshBundle, registry::BlockRegistry};

pub struct World {
    seed: i32,
    reg: Arc<BlockRegistry>,
    gen: Arc<TerrainGenerator>,
    chunks: RwLock<HashMap<ChunkPos, Chunk>>,
    meshes: RwLock<HashMap<ChunkPos, ChunkMeshBundle>>,
    sun_dir: RwLock<glam::Vec3>,
    /// Positions of water sources/flowing water still spreading. Drained and
    /// refilled each water tick by `tick_water`.
    pending_flow: RwLock<HashSet<IVec3>>,
    /// Set of positions that are water sources (level 8). Used for O(1)
    /// lookups during water simulation instead of scanning all chunks.
    source_water: RwLock<HashSet<IVec3>>,
    /// Accumulated wall-clock seconds since the last water tick.
    water_tick_accumulator: RwLock<f32>,
    /// Self-reference for safe cross-thread closure capture. Built once in
    /// `new_with_path` via `Arc::new_cyclic`; used by `with_chunk_for_mesh`,
    /// `recompute_lighting_at`, and `set_block` to hand closures an `Arc<World>`
    /// without using raw `*const Self` pointer aliasing.
    self_ref: Weak<World>,
}

/// Seconds between water simulation steps. Minecraft ticks water every 5 game
/// ticks (~0.25s at 20 TPS).
pub const WATER_TICK_INTERVAL: f32 = 0.25;

impl World {
    pub fn new(seed: i32) -> Arc<Self> {
        Self::new_with_path(seed, None)
    }

    /// Create a world, optionally loading block definitions from a JSON file
    /// at `assets_path/blocks/blocks.json`. Falls back to built-in blocks if
    /// the path is `None` or the file doesn't exist.
    pub fn new_with_path(seed: i32, assets_path: Option<&std::path::Path>) -> Arc<Self> {
        let reg = match assets_path {
            Some(path) => {
                let loader = voxel_assets::AssetLoader::new(path);
                match loader.load_blocks() {
                    Ok(blocks) => {
                        log::info!("loaded {} blocks from {}", blocks.len(), path.display());
                        Arc::new(BlockRegistry::from_assets(&blocks))
                    }
                    Err(e) => {
                        log::warn!(
                            "failed to load blocks from {}: {e}. Using builtins.",
                            path.display()
                        );
                        Arc::new(BlockRegistry::with_builtins())
                    }
                }
            }
            None => Arc::new(BlockRegistry::with_builtins()),
        };
        let gen = Arc::new(TerrainGenerator::new(seed));
        // `Arc::new_cyclic` lets the struct hold a `Weak<Self>` back-reference
        // without a chicken-and-egg bootstrap. The closure receives the Weak
        // during construction; we close over it to seed `self_ref`.
        Arc::new_cyclic(|weak: &Weak<Self>| Self {
            seed,
            reg,
            gen,
            chunks: RwLock::new(HashMap::new()),
            meshes: RwLock::new(HashMap::new()),
            sun_dir: RwLock::new(glam::Vec3::new(0.3, 0.9, 0.1).normalize()),
            pending_flow: RwLock::new(HashSet::new()),
            source_water: RwLock::new(HashSet::new()),
            water_tick_accumulator: RwLock::new(0.0),
            self_ref: Weak::clone(weak),
        })
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }
    pub fn registry(&self) -> Arc<BlockRegistry> {
        self.reg.clone()
    }
    pub fn terrain(&self) -> Arc<TerrainGenerator> {
        self.gen.clone()
    }

    pub fn set_sun_dir(&self, dir: glam::Vec3) {
        *self.sun_dir.write() = dir;
    }

    pub fn get_sun_dir(&self) -> glam::Vec3 {
        *self.sun_dir.read()
    }

    // --- chunk storage -----------------------------------------------------

    pub fn insert_chunk(&self, pos: ChunkPos, mut chunk: Chunk) {
        chunk.pos = pos;
        self.chunks.write().insert(pos, chunk);
    }

    pub fn remove_chunk(&self, pos: ChunkPos) {
        let mut chunks = self.chunks.write();
        let mut meshes = self.meshes.write();
        chunks.remove(&pos);
        meshes.remove(&pos);
    }

    pub fn insert_mesh(&self, pos: ChunkPos, bundle: ChunkMeshBundle) {
        self.meshes.write().insert(pos, bundle);
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.read().len()
    }
    pub fn meshed_chunk_count(&self) -> usize {
        self.meshes.read().len()
    }

    /// Get all loaded chunks (for save).
    pub fn all_loaded_chunks(&self) -> Vec<(ChunkPos, Chunk)> {
        self.chunks
            .read()
            .iter()
            .map(|(&cp, c)| (cp, c.clone()))
            .collect()
    }

    /// Insert multiple chunks (for load).
    pub fn insert_chunks(&self, chunks: Vec<(ChunkPos, Chunk)>) {
        let mut map = self.chunks.write();
        for (cp, mut chunk) in chunks {
            chunk.pos = cp;
            map.insert(cp, chunk);
        }
    }

    /// Get sunlight at a world coordinate (0 if chunk not loaded).
    pub fn get_sunlight_world(&self, x: i32, y: i32, z: i32) -> u8 {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return 0;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read();
        let Some(chunk) = chunks.get(&cp) else {
            if y >= voxel_core::SEA_LEVEL {
                return 15;
            }
            return 0;
        };
        let origin = chunk_origin(cp);
        chunk.get_sunlight(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Get torchlight at a world coordinate (0 if chunk not loaded).
    pub fn get_torchlight_world(&self, x: i32, y: i32, z: i32) -> u8 {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return 0;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read();
        let Some(chunk) = chunks.get(&cp) else {
            return 0;
        };
        let origin = chunk_origin(cp);
        chunk.get_torchlight(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Set torchlight at a world coordinate (no-op if chunk not loaded).
    pub fn set_torchlight_world(&self, x: i32, y: i32, z: i32, v: u8) {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let mut chunks = self.chunks.write();
        let Some(chunk) = chunks.get_mut(&cp) else {
            return;
        };
        let origin = chunk_origin(cp);
        chunk.set_torchlight(x - origin.x, y - origin.y, z - origin.z, v);
    }

    /// Recompute sunlight and torchlight for a single chunk. Used after water
    /// flow simulation modifies block types/light absorption across chunks.
    pub fn recompute_lighting_at(&self, cp: ChunkPos) {
        let reg = self.reg.clone();
        let chunks = self.chunks.read();
        let Some(chunk) = chunks.get(&cp) else {
            return;
        };
        let mut chunk_copy = chunk.clone();
        drop(chunks);

        let sun_dir = self.get_sun_dir();

        // Upgrade the self-reference once per call (`self_ref` was seeded via
        // `Arc::new_cyclic`); closures capture the resulting `Arc<World>` by
        // move so cross-thread access is safe without raw self-pointers.
        let arc = self.self_ref.upgrade().expect("World outlives use");
        let sample_block = {
            let arc = Arc::clone(&arc);
            move |wx: i32, wy: i32, wz: i32| arc.get_block(wx, wy, wz)
        };
        let sample_torch = {
            let arc = Arc::clone(&arc);
            move |wx: i32, wy: i32, wz: i32| arc.get_torchlight_world(wx, wy, wz)
        };

        let mut cross_updates = Vec::new();
        crate::light::compute_all(
            &mut chunk_copy,
            &reg,
            sun_dir,
            &sample_block,
            &sample_torch,
            &mut |pos, level| cross_updates.push((pos, level)),
        );

        for (pos, level) in cross_updates {
            self.set_torchlight_world(pos.0.x, pos.0.y, pos.0.z, level);
        }

        let mut chunks = self.chunks.write();
        if let Some(chunk) = chunks.get_mut(&cp) {
            chunk.sunlight = chunk_copy.sunlight.clone();
            chunk.torchlight = chunk_copy.torchlight.clone();
            chunk.dirty = true;
            chunk.light_dirty = true;
        }
    }

    /// True if a chunk is loaded (generated) at the given chunk position.
    pub fn is_chunk_loaded(&self, pos: ChunkPos) -> bool {
        self.chunks.read().contains_key(&pos)
    }

    /// Chunk debug info for the minimap visualization.
    /// Returns (loaded, dirty, palette_mode, has_mesh).
    pub fn chunk_debug_info(&self, pos: ChunkPos) -> (bool, bool, bool, bool) {
        let loaded = {
            let chunks = self.chunks.read();
            match chunks.get(&pos) {
                Some(c) => (true, c.dirty, c.is_palette_mode()),
                None => return (false, false, false, false),
            }
        };
        let has_mesh = self.meshes.read().contains_key(&pos);
        (loaded.0, loaded.1, loaded.2, has_mesh)
    }

    /// Batch version: acquires chunks + meshes locks once for an entire minimap grid.
    pub fn chunk_debug_info_batch(
        &self,
        center: ChunkPos,
        half: i32,
    ) -> Vec<(ChunkPos, bool, bool, bool, bool)> {
        let mut result = Vec::with_capacity(((half * 2 + 1) * (half * 2 + 1)) as usize);
        let chunks = self.chunks.read();
        let meshes = self.meshes.read();
        for dx in -half..=half {
            for dz in -half..=half {
                let pos = ChunkPos::new(center.x() + dx, 0, center.z() + dz);
                let (loaded, dirty, palette_mode) = match chunks.get(&pos) {
                    Some(c) => (true, c.dirty, c.is_palette_mode()),
                    None => (false, false, false),
                };
                let has_mesh = meshes.contains_key(&pos);
                result.push((pos, loaded, dirty, palette_mode, has_mesh));
            }
        }
        result
    }

    /// True if the chunk containing the given world block position is loaded.
    pub fn is_block_loaded(&self, x: i32, y: i32, z: i32) -> bool {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return false;
        }
        self.is_chunk_loaded(block_to_chunk(IVec3::new(x, y, z)))
    }

    /// All currently-meshed chunk positions + a cheap clone of each bundle, for
    /// the renderer to sync GPU buffers. Clones are shallow-ish (Vec of PODs).
    pub fn snapshot_meshes(&self) -> Vec<(ChunkPos, ChunkMeshBundle)> {
        self.meshes
            .read()
            .iter()
            .map(|(p, b)| (*p, b.clone()))
            .collect()
    }

    // --- block queries -----------------------------------------------------

    /// Get a block by world coordinate. Returns air for unloaded chunks or Y
    /// out of range (so the world looks "empty" where data isn't loaded yet).
    pub fn get_block(&self, x: i32, y: i32, z: i32) -> BlockId {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return BlockId::AIR;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read();
        let Some(chunk) = chunks.get(&cp) else {
            return BlockId::AIR;
        };
        let origin = chunk_origin(cp);
        chunk.get(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Get a read reference to the chunks map for batch access.
    /// The caller must not hold this longer than necessary to avoid blocking writers.
    pub fn chunks_ref(&self) -> &parking_lot::RwLock<std::collections::HashMap<ChunkPos, Chunk>> {
        &self.chunks
    }

    /// Get a reference to the block registry.
    pub fn registry_ref(&self) -> &BlockRegistry {
        &self.reg
    }
    #[inline]
    pub fn get_block_guarded(
        chunks: &std::collections::HashMap<ChunkPos, Chunk>,
        x: i32,
        y: i32,
        z: i32,
    ) -> BlockId {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return BlockId::AIR;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let Some(chunk) = chunks.get(&cp) else {
            return BlockId::AIR;
        };
        let origin = chunk_origin(cp);
        chunk.get(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Check solidity using a pre-acquired chunks read guard.
    #[inline]
    pub fn is_solid_guarded(
        chunks: &std::collections::HashMap<ChunkPos, Chunk>,
        reg: &BlockRegistry,
        x: i32,
        y: i32,
        z: i32,
    ) -> bool {
        let id = Self::get_block_guarded(chunks, x, y, z);
        reg.is_solid(id)
    }

    /// Set a block by world coordinate. Returns true if a loaded chunk was
    /// updated. Also recomputes lighting for the affected column + torchlight.
    pub fn set_block(&self, x: i32, y: i32, z: i32, id: BlockId) -> bool {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return false;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let origin = chunk_origin(cp);
        let lx = x - origin.x;
        let ly = y - origin.y;
        let lz = z - origin.z;

        // Get the chunk write lock, set the block, then run lighting.
        {
            let mut chunks = self.chunks.write();
            let Some(chunk) = chunks.get_mut(&cp) else {
                return false;
            };
            chunk.set(lx, ly, lz, id);
        }

        // Recompute lighting for this chunk using the new ray-based system.
        let reg = self.reg.clone();
        let chunks = self.chunks.read();
        if let Some(chunk) = chunks.get(&cp) {
            let mut chunk_copy = chunk.clone();
            drop(chunks);

            // Sun direction for shadows — read from world state.
            let sun_dir = self.get_sun_dir();

            // Upgrade the self-reference once per call; closures capture
            // `Arc<World>` by move so cross-thread access is safe without raw
            // self-pointers (see Design B in the refactor notes).
            let arc = self.self_ref.upgrade().expect("World outlives use");
            let sample_block = {
                let arc = Arc::clone(&arc);
                move |wx: i32, wy: i32, wz: i32| arc.get_block(wx, wy, wz)
            };
            let sample_torch = {
                let arc = Arc::clone(&arc);
                move |wx: i32, wy: i32, wz: i32| arc.get_torchlight_world(wx, wy, wz)
            };

            let mut cross_updates = Vec::new();
            crate::light::compute_all(
                &mut chunk_copy,
                &reg,
                sun_dir,
                &sample_block,
                &sample_torch,
                &mut |pos, level| cross_updates.push((pos, level)),
            );

            // Apply cross-chunk torchlight updates.
            for (pos, level) in cross_updates {
                self.set_torchlight_world(pos.0.x, pos.0.y, pos.0.z, level);
            }

            // Write the relit chunk back.
            let mut chunks = self.chunks.write();
            if let Some(chunk) = chunks.get_mut(&cp) {
                chunk.sunlight = chunk_copy.sunlight.clone();
                chunk.torchlight = chunk_copy.torchlight.clone();
                chunk.light_dirty = true;
            }
        }
        // NOTE: water flow simulation is NOT called here because the water
        // level may not be set yet (bucket places block then sets level).
        // Callers that need flow should enqueue the position (via
        // `place_water` / `remove_water`) and let `tick_water` drive the
        // incremental spread.

        true
    }

    /// Convenience: set a block and report the owning chunk position, if any.
    pub fn set_block_world(&self, pos: BlockPos, id: BlockId) -> Option<ChunkPos> {
        if self.set_block(pos.0.x, pos.0.y, pos.0.z, id) {
            Some(block_to_chunk(pos.0))
        } else {
            None
        }
    }

    /// True if the block at (x,y,z) is collidable per the registry.
    pub fn is_solid(&self, x: i32, y: i32, z: i32) -> bool {
        let id = self.get_block(x, y, z);
        self.reg.is_solid(id)
    }

    // --- water queries ---------------------------------------------------

    /// Get water level (0–8) at a world coordinate. Returns 0 if not liquid.
    pub fn get_water_level_world(&self, x: i32, y: i32, z: i32) -> u8 {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return 0;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let chunks = self.chunks.read();
        let Some(chunk) = chunks.get(&cp) else {
            return 0;
        };
        let origin = chunk_origin(cp);
        chunk.get_water_level(x - origin.x, y - origin.y, z - origin.z)
    }

    /// Set water level (0–8) at a world coordinate. No-op if chunk not loaded.
    /// Also sets the block to water if level > 0, or air if level == 0.
    pub fn set_water_level_world(&self, x: i32, y: i32, z: i32, level: u8) {
        if !(0..WORLD_HEIGHT_BLOCKS).contains(&y) {
            return;
        }
        let cp = block_to_chunk(IVec3::new(x, y, z));
        let mut chunks = self.chunks.write();
        let Some(chunk) = chunks.get_mut(&cp) else {
            return;
        };
        let origin = chunk_origin(cp);
        let lx = x - origin.x;
        let ly = y - origin.y;
        let lz = z - origin.z;

        if level > 0 {
            let water_id = self.reg.id_of("water").unwrap();
            let current = chunk.get(lx, ly, lz);
            if current.is_air() || self.reg.is_liquid(current) {
                if current != water_id {
                    chunk.set(lx, ly, lz, water_id);
                }
            }
        } else {
            let current = chunk.get(lx, ly, lz);
            if self.reg.is_liquid(current) {
                chunk.set(lx, ly, lz, BlockId::AIR);
            }
        }
        chunk.set_water_level(lx, ly, lz, level);
    }

    /// True if the block at (x,y,z) is a water source (liquid with level 8).
    /// Uses the source index for O(1) lookup.
    pub fn is_water_source(&self, x: i32, y: i32, z: i32) -> bool {
        self.is_known_water_source(x, y, z)
    }

    /// True if the block at (x,y,z) is any liquid.
    pub fn is_liquid(&self, x: i32, y: i32, z: i32) -> bool {
        let id = self.get_block(x, y, z);
        self.reg.is_liquid(id)
    }

    /// Get the set of all water source positions (read access).
    pub fn water_sources(&self) -> &RwLock<HashSet<IVec3>> {
        &self.source_water
    }

    /// Check if a position is a known water source (O(1) via index).
    pub fn is_known_water_source(&self, x: i32, y: i32, z: i32) -> bool {
        self.source_water.read().contains(&IVec3::new(x, y, z))
    }

    /// Remove a water source block at (x,y,z). Uses `set_block` to set air
    /// so lighting is properly recalculated, then enqueues neighbouring water
    /// positions so the simulation resumes on the next tick. Returns true if
    /// removed.
    pub fn remove_water(&self, x: i32, y: i32, z: i32) -> bool {
        if !self.is_water_source(x, y, z) {
            return false;
        }
        // Use set_block to remove water so lighting is recalculated.
        self.set_block(x, y, z, BlockId::AIR);
        // Clear the water level array too (set_block only changes block ID).
        self.set_water_level_world(x, y, z, 0);
        // Remove from the source index so subsequent lookups don't see it.
        self.source_water.write().remove(&IVec3::new(x, y, z));
        // Remove from pending flow and enqueue neighbours so the surrounding
        // water resumes spreading on the next tick.
        {
            let mut pending = self.pending_flow.write();
            pending.remove(&IVec3::new(x, y, z));
            for npos in water_neighbours(IVec3::new(x, y, z)) {
                if self.is_block_loaded(npos.x, npos.y, npos.z)
                    && self.reg.is_liquid(self.get_block(npos.x, npos.y, npos.z))
                {
                    pending.insert(npos);
                }
            }
        }
        true
    }

    /// Place a water source block at (x,y,z). Sets the block + water level 8
    /// and enqueues the position for the simulation. Spread happens over
    /// subsequent ticks via `tick_water` (not synchronously here).
    pub fn place_water(&self, x: i32, y: i32, z: i32) -> bool {
        let water_id = match self.reg.id_of("water") {
            Some(id) => id,
            None => return false,
        };
        if !self.set_block(x, y, z, water_id) {
            return false;
        }
        // Set water level to 8 (source) after block is placed.
        self.set_water_level_world(x, y, z, 8);
        // Enqueue the source for incremental flow on the next tick.
        self.pending_flow.write().insert(IVec3::new(x, y, z));
        // Track the source position in the O(1) index for fast lookups.
        self.source_water.write().insert(IVec3::new(x, y, z));
        true
    }

    /// Advance the water simulation by `dt` seconds. When the internal
    /// accumulator crosses `WATER_TICK_INTERVAL`, runs one flow step and
    /// returns the chunks modified (so the caller can request remeshes).
    /// Lighting is intentionally NOT recomputed here — water level changes
    /// don't meaningfully alter light absorption, and the cost would add up
    /// for large water regions.
    pub fn tick_water(&self, dt: f32) -> HashSet<ChunkPos> {
        {
            let mut acc = self.water_tick_accumulator.write();
            *acc += dt;
            if *acc < WATER_TICK_INTERVAL {
                return HashSet::new();
            }
            *acc -= WATER_TICK_INTERVAL;
            // Clamp to avoid runaway after long pauses.
            if *acc > WATER_TICK_INTERVAL {
                *acc = WATER_TICK_INTERVAL;
            }
        }
        let mut pending = self.pending_flow.write();
        crate::water::simulate_flow_step(self, &mut pending)
    }

    // --- meshing support ---------------------------------------------------

    /// Run a closure with read access to a chunk and a neighbour-sampling
    /// function that crosses chunk borders (used by the mesher on worker
    /// threads). The samplers read through the shared `RwLock`. The water
    /// sampler returns the level (0-8) at the given world coordinate, or 0
    /// for non-water / unloaded positions; it is used by the mesher to fill
    /// the "step" between adjacent water layers at different levels. The
    /// loaded sampler returns true when the chunk at the given world
    /// coordinate is loaded; the mesher uses it to decide whether to apply
    /// the chunk-border face-ownership rule (which prevents Z-fighting)
    /// without leaving holes at the edge of the loaded area.
    pub fn with_chunk_for_mesh<R>(
        &self,
        pos: ChunkPos,
        f: impl FnOnce(
            &Chunk,
            &dyn Fn(i32, i32, i32) -> BlockId,
            &dyn Fn(i32, i32, i32) -> u8,
            &dyn Fn(i32, i32, i32) -> bool,
        ) -> R,
    ) -> R {
        // Clone the target chunk out from under the lock so meshing is lock-free;
        // neighbours are sampled live (cheap read lock per sample is acceptable).
        let chunk = {
            let chunks = self.chunks.read();
            chunks.get(&pos).cloned()
        };

        // Upgrade the self-reference once per call. The captured `Arc<World>`
        // is what keeps the World alive across the closure lifetime; each
        // sampler clones it independently so the closures don't share state.
        let world_arc = self
            .self_ref
            .upgrade()
            .expect("`World` outlives `with_chunk_for_mesh` callers");

        let Some(chunk) = chunk else {
            // No chunk: produce an empty result by giving the closure an empty
            // chunk. This path should be rare (mesh requested for unloaded chunk).
            let empty = Chunk::new(pos);
            let sample: Box<dyn Fn(i32, i32, i32) -> BlockId> = Box::new(|_, _, _| BlockId::AIR);
            let sample_water: Box<dyn Fn(i32, i32, i32) -> u8> = Box::new(|_, _, _| 0);
            let sample_loaded: Box<dyn Fn(i32, i32, i32) -> bool> = Box::new(|_, _, _| false);
            return f(&empty, &*sample, &*sample_water, &*sample_loaded);
        };

        let sample: Box<dyn Fn(i32, i32, i32) -> BlockId> = {
            let arc = Arc::clone(&world_arc);
            Box::new(move |x, y, z| arc.get_block(x, y, z))
        };
        let sample_water: Box<dyn Fn(i32, i32, i32) -> u8> = {
            let arc = Arc::clone(&world_arc);
            Box::new(move |x, y, z| arc.get_water_level_world(x, y, z))
        };
        let sample_loaded: Box<dyn Fn(i32, i32, i32) -> bool> = {
            let arc = Arc::clone(&world_arc);
            Box::new(move |x, y, z| arc.is_block_loaded(x, y, z))
        };
        f(&chunk, &*sample, &*sample_water, &*sample_loaded)
    }
}

/// Four cardinal neighbour offsets of `pos` on the same Y plane.
fn water_neighbours(pos: IVec3) -> [IVec3; 4] {
    [
        pos + IVec3::new(1, 0, 0),
        pos + IVec3::new(-1, 0, 0),
        pos + IVec3::new(0, 0, 1),
        pos + IVec3::new(0, 0, -1),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Chunk;
    use voxel_core::BlockId;

    #[test]
    fn set_block_returns_false_for_unloaded_chunk() {
        let world = World::new(42);
        // No chunks loaded.
        assert!(!world.set_block(0, 0, 0, BlockId(1)));
    }

    #[test]
    fn get_block_returns_air_for_unloaded() {
        let world = World::new(42);
        assert_eq!(world.get_block(0, 0, 0), BlockId::AIR);
    }

    #[test]
    fn get_block_out_of_y_range_returns_air() {
        let world = World::new(42);
        assert_eq!(world.get_block(0, 1000, 0), BlockId::AIR);
        assert_eq!(world.get_block(0, -1, 0), BlockId::AIR);
    }

    #[test]
    fn is_block_loaded_false_for_unloaded() {
        let world = World::new(42);
        assert!(!world.is_block_loaded(0, 0, 0));
    }

    #[test]
    fn insert_and_remove_chunk() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let chunk = Chunk::new(cp);
        world.insert_chunk(cp, chunk);
        assert!(world.is_chunk_loaded(cp));
        assert_eq!(world.loaded_chunk_count(), 1);
        world.remove_chunk(cp);
        assert!(!world.is_chunk_loaded(cp));
        assert_eq!(world.loaded_chunk_count(), 0);
    }

    #[test]
    fn set_block_loaded_chunk() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let chunk = Chunk::new(cp);
        world.insert_chunk(cp, chunk);
        assert!(world.set_block(0, 0, 0, BlockId(2)));
        assert_eq!(world.get_block(0, 0, 0), BlockId(2));
    }

    #[test]
    fn is_solid_uses_registry() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        let stone = world.registry().id_of("stone").unwrap();
        chunk.set(0, 0, 0, stone);
        world.insert_chunk(cp, chunk);
        assert!(world.is_solid(0, 0, 0));
        assert!(!world.is_solid(1, 0, 0));
    }

    #[test]
    fn get_torchlight_unloaded_returns_zero() {
        let world = World::new(42);
        assert_eq!(world.get_torchlight_world(0, 0, 0), 0);
    }

    #[test]
    fn set_torchlight_noop_for_unloaded() {
        let world = World::new(42);
        world.set_torchlight_world(0, 0, 0, 10);
        // No panic, no effect.
        assert_eq!(world.get_torchlight_world(0, 0, 0), 0);
    }

    #[test]
    fn chunk_debug_info_unloaded() {
        let world = World::new(42);
        let info = world.chunk_debug_info(ChunkPos::new(0, 0, 0));
        assert!(!info.0);
    }

    #[test]
    fn snapshot_meshes_empty() {
        let world = World::new(42);
        assert!(world.snapshot_meshes().is_empty());
    }

    #[test]
    fn all_loaded_chunks_roundtrip() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        chunk.set(0, 0, 0, BlockId(5));
        world.insert_chunk(cp, chunk);
        let chunks = world.all_loaded_chunks();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].0, cp);
        assert_eq!(chunks[0].1.get(0, 0, 0), BlockId(5));
    }

    #[test]
    fn water_source_index_tracking() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        // Place water — should add to index.
        world.place_water(1, 10, 1);
        assert!(world.is_known_water_source(1, 10, 1));
        assert_eq!(world.water_sources().read().len(), 1);

        // Remove water — should remove from index.
        world.remove_water(1, 10, 1);
        assert!(!world.is_known_water_source(1, 10, 1));
        assert_eq!(world.water_sources().read().len(), 0);
    }

    #[test]
    fn water_source_index_multiple() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        world.place_water(0, 5, 0);
        world.place_water(1, 5, 0);
        world.place_water(2, 5, 0);
        assert_eq!(world.water_sources().read().len(), 3);
        assert!(world.is_known_water_source(0, 5, 0));
        assert!(world.is_known_water_source(1, 5, 0));
        assert!(world.is_known_water_source(2, 5, 0));

        // Remove one — index should shrink to 2.
        world.remove_water(1, 5, 0);
        assert_eq!(world.water_sources().read().len(), 2);
        assert!(!world.is_known_water_source(1, 5, 0));
        assert!(world.is_known_water_source(0, 5, 0));
        assert!(world.is_known_water_source(2, 5, 0));
    }

    #[test]
    fn water_source_index_idempotent_place() {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        world.insert_chunk(cp, Chunk::new(cp));

        // Place water twice at the same position — index should still have 1.
        world.place_water(0, 5, 0);
        world.place_water(0, 5, 0);
        assert_eq!(world.water_sources().read().len(), 1);
    }
}
