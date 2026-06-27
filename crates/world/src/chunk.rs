//! Chunk storage: a 16³ block volume with palette-based compression.
//!
//! Blocks are stored using a palette + index mapping:
//! - A palette of unique `BlockId` values (max 256 for u8 indices)
//! - Per-block u8 index into the palette
//! - Falls back to flat `Box<[BlockId]>` storage when palette overflows
//!
//! Layout is y-major, z-middle, x-fastest so that vertical column scans (used
//! during worldgen and meshing) are cache-friendly. Indexing helpers live in
//! `voxel_core::math`.

use voxel_core::{BlockId, ChunkPos, CHUNK_CUBED};

use crate::registry::BlockRegistry;

/// One cubic chunk section: 16×16×16 blocks.
#[derive(Clone)]
pub struct Chunk {
    pub pos: ChunkPos,
    /// Palette: unique block IDs used in this chunk. Empty when in flat mode.
    palette: Vec<BlockId>,
    /// Index array: for each block position, the index into `palette`.
    /// Only allocated when in palette mode.
    indices: Box<[u8]>,
    /// Flat block storage. Used when palette exceeds 255 unique blocks,
    /// or as a cache when fast slice access is needed.
    flat: Box<[BlockId]>,
    /// True when using palette mode (indices is valid).
    palette_mode: bool,
    /// Sunlight level 0–15 per block.
    pub(crate) sunlight: Box<[u8]>,
    /// Torchlight level 0–15 per block.
    pub(crate) torchlight: Box<[u8]>,
    /// Water level 0–8 per block. 0 = no water, 1 = shallowest flow, 8 = source.
    pub(crate) water_level: Box<[u8]>,
    /// Set whenever a block changes so the mesher/streamer knows to remesh.
    pub dirty: bool,
    /// Set whenever lighting changes and needs recomputation.
    pub light_dirty: bool,
    /// True once `TerrainGenerator` has populated this chunk.
    pub generated: bool,
}

impl Chunk {
    pub fn new(pos: ChunkPos) -> Self {
        Self {
            pos,
            palette: Vec::new(),
            indices: Box::new([0u8; CHUNK_CUBED]),
            flat: vec![BlockId::AIR; CHUNK_CUBED].into_boxed_slice(),
            palette_mode: false,
            sunlight: vec![0u8; CHUNK_CUBED].into_boxed_slice(),
            torchlight: vec![0u8; CHUNK_CUBED].into_boxed_slice(),
            water_level: vec![0u8; CHUNK_CUBED].into_boxed_slice(),
            dirty: false,
            light_dirty: true,
            generated: false,
        }
    }

    #[inline]
    pub fn blocks(&self) -> &[BlockId] {
        &self.flat
    }

    /// Get a block by local coordinate. Returns air for out-of-range locals.
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> BlockId {
        if !in_bounds(x, y, z) {
            return BlockId::AIR;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        if self.palette_mode {
            let pal_idx = self.indices[idx] as usize;
            self.palette[pal_idx]
        } else {
            self.flat[idx]
        }
    }

    /// Set a block by local coordinate. Ignores out-of-range writes and marks
    /// the chunk dirty on change.
    #[inline]
    pub fn set(&mut self, x: i32, y: i32, z: i32, id: BlockId) {
        if !in_bounds(x, y, z) {
            return;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        if self.palette_mode {
            // Check if already set to this value.
            let pal_idx = self.indices[idx] as usize;
            if self.palette[pal_idx] == id {
                return;
            }
            // Try to find or add to palette.
            if let Some(new_pal_idx) = self.find_or_add_palette(id) {
                self.indices[idx] = new_pal_idx as u8;
            } else {
                // Palette overflow: convert to flat mode.
                self.convert_to_flat();
                self.flat[idx] = id;
            }
        } else {
            if self.flat[idx] == id {
                return;
            }
            self.flat[idx] = id;
            // Try to switch to palette mode if beneficial.
            self.consider_palette_mode();
        }
        self.dirty = true;
    }

    /// Find `id` in the palette, or add it. Returns the palette index, or None
    /// if palette is full (>255 unique blocks).
    fn find_or_add_palette(&mut self, id: BlockId) -> Option<usize> {
        if let Some(pos) = self.palette.iter().position(|&p| p == id) {
            return Some(pos);
        }
        if self.palette.len() >= 256 {
            return None;
        }
        self.palette.push(id);
        Some(self.palette.len() - 1)
    }

    /// After enough unique blocks accumulate in flat mode, switch to palette mode.
    fn consider_palette_mode(&mut self) {
        if self.palette_mode {
            return;
        }
        // Count unique blocks. If <= 128, switch to palette mode.
        let mut unique = std::collections::HashSet::with_capacity(128);
        let mut unique_vec = Vec::with_capacity(128);
        for &b in self.flat.iter() {
            if unique.insert(b) {
                unique_vec.push(b);
                if unique_vec.len() > 128 {
                    return; // Too many unique blocks, stay in flat mode.
                }
            }
        }
        // Switch to palette mode.
        self.palette = unique_vec;
        let mut lookup: std::collections::HashMap<BlockId, u8> = std::collections::HashMap::new();
        for (i, &p) in self.palette.iter().enumerate() {
            lookup.insert(p, i as u8);
        }
        for i in 0..CHUNK_CUBED {
            let id = self.flat[i];
            let pal_idx = *lookup.get(&id).unwrap();
            self.indices[i] = pal_idx;
        }
        self.palette_mode = true;
    }

    /// Convert from palette mode to flat mode.
    fn convert_to_flat(&mut self) {
        if !self.palette_mode {
            return;
        }
        for i in 0..CHUNK_CUBED {
            self.flat[i] = self.palette[self.indices[i] as usize];
        }
        self.palette.clear();
        self.palette.shrink_to_fit();
        self.palette_mode = false;
    }

    /// Get sunlight (0–15) at local coords. Returns 0 out of bounds.
    #[inline]
    pub fn get_sunlight(&self, x: i32, y: i32, z: i32) -> u8 {
        if !in_bounds(x, y, z) {
            return 0;
        }
        self.sunlight[voxel_core::math::local_index(x, y, z)]
    }

    /// Set sunlight (0–15) at local coords. Ignores out of bounds.
    #[inline]
    pub fn set_sunlight(&mut self, x: i32, y: i32, z: i32, v: u8) {
        if !in_bounds(x, y, z) {
            return;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        if self.sunlight[idx] != v {
            self.sunlight[idx] = v;
            self.light_dirty = true;
        }
    }

    /// Get torchlight (0–15) at local coords. Returns 0 out of bounds.
    #[inline]
    pub fn get_torchlight(&self, x: i32, y: i32, z: i32) -> u8 {
        if !in_bounds(x, y, z) {
            return 0;
        }
        self.torchlight[voxel_core::math::local_index(x, y, z)]
    }

    /// Set torchlight (0–15) at local coords. Ignores out of bounds.
    #[inline]
    pub fn set_torchlight(&mut self, x: i32, y: i32, z: i32, v: u8) {
        if !in_bounds(x, y, z) {
            return;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        if self.torchlight[idx] != v {
            self.torchlight[idx] = v;
            self.light_dirty = true;
        }
    }

    /// Get water level (0–8) at local coords. 0 = no water, 8 = source.
    #[inline]
    pub fn get_water_level(&self, x: i32, y: i32, z: i32) -> u8 {
        if !in_bounds(x, y, z) {
            return 0;
        }
        self.water_level[voxel_core::math::local_index(x, y, z)]
    }

    /// Set water level (0–8) at local coords. Ignores out of bounds.
    #[inline]
    pub fn set_water_level(&mut self, x: i32, y: i32, z: i32, v: u8) {
        if !in_bounds(x, y, z) {
            return;
        }
        let idx = voxel_core::math::local_index(x, y, z);
        if self.water_level[idx] != v {
            self.water_level[idx] = v;
            self.dirty = true;
            self.light_dirty = true; // Water changes light absorption.
        }
    }

    /// Combined light = max(sunlight, torchlight), used by the mesher.
    #[inline]
    pub fn get_combined_light(&self, x: i32, y: i32, z: i32) -> u8 {
        self.get_sunlight(x, y, z).max(self.get_torchlight(x, y, z))
    }

    /// Reset all light to 0 (call before recomputing).
    pub fn clear_light(&mut self) {
        for s in self.sunlight.iter_mut() {
            *s = 0;
        }
        for t in self.torchlight.iter_mut() {
            *t = 0;
        }
        self.light_dirty = true;
    }

    /// Count non-air blocks (used for empty-chunk culling in the mesher).
    pub fn non_air_count(&self) -> usize {
        if self.palette_mode {
            // Fast path: check if air is in the palette.
            if let Some(air_idx) = self.palette.iter().position(|b| b.is_air()) {
                let air_idx = air_idx as u8;
                self.indices.iter().filter(|&&i| i != air_idx).count()
            } else {
                // No air in palette, all blocks are non-air.
                CHUNK_CUBED
            }
        } else {
            self.flat.iter().filter(|b| !b.is_air()).count()
        }
    }

    /// Highest non-air block in a local column `(x, z)`, or 0 if empty.
    pub fn column_height(&self, x: i32, z: i32) -> i32 {
        if !in_bounds(x, 0, z) {
            return 0;
        }
        for y in (0..voxel_core::CHUNK_SIZE).rev() {
            if !self.get(x, y, z).is_air() {
                return y + 1;
            }
        }
        0
    }

    /// Fill the chunk from a function keyed by local coordinate. Used by gen.
    pub fn fill_with(&mut self, mut f: impl FnMut(i32, i32, i32) -> BlockId) {
        // Always use flat mode for fill_with to avoid repeated palette lookups.
        self.convert_to_flat();
        for y in 0..voxel_core::CHUNK_SIZE {
            for z in 0..voxel_core::CHUNK_SIZE {
                for x in 0..voxel_core::CHUNK_SIZE {
                    let id = f(x, y, z);
                    let idx = voxel_core::math::local_index(x, y, z);
                    self.flat[idx] = id;
                }
            }
        }
        self.dirty = true;
        self.consider_palette_mode();
    }

    /// Convenience: replace every block with `id`.
    pub fn fill_uniform(&mut self, id: BlockId) {
        if self.palette_mode {
            // Just reset palette to single entry.
            self.palette.clear();
            self.palette.push(id);
            for i in self.indices.iter_mut() {
                *i = 0;
            }
        } else {
            for b in self.flat.iter_mut() {
                *b = id;
            }
            self.consider_palette_mode();
        }
        self.dirty = true;
    }

    /// Number of solid (collidable) blocks per the given registry.
    pub fn solid_count(&self, reg: &BlockRegistry) -> usize {
        if self.palette_mode {
            // Pre-compute solidity per palette entry.
            let solid: Vec<bool> = self.palette.iter().map(|b| reg.is_solid(*b)).collect();
            self.indices.iter().filter(|&&i| solid[i as usize]).count()
        } else {
            self.flat.iter().filter(|b| reg.is_solid(**b)).count()
        }
    }

    /// Returns true if the chunk is in palette mode (for diagnostics).
    pub fn is_palette_mode(&self) -> bool {
        self.palette_mode
    }

    /// Number of unique blocks in the palette (0 if in flat mode).
    pub fn palette_len(&self) -> usize {
        if self.palette_mode {
            self.palette.len()
        } else {
            0
        }
    }

    // --- Save/Load accessors ---

    /// Get palette data (for save).
    pub fn palette_data(&self) -> &[BlockId] {
        &self.palette
    }

    /// Get index data (for save).
    pub fn indices_data(&self) -> &[u8] {
        &self.indices
    }

    /// Get sunlight data (for save).
    pub fn sunlight_data(&self) -> &[u8] {
        &self.sunlight
    }

    /// Get torchlight data (for save).
    pub fn torchlight_data(&self) -> &[u8] {
        &self.torchlight
    }

    /// Get water level data (for save).
    pub fn water_level_data(&self) -> &[u8] {
        &self.water_level
    }

    /// Restore from palette + indices (for load).
    pub fn restore_palette(&mut self, palette: Vec<BlockId>, indices: Vec<u8>) {
        self.palette = palette;
        self.indices = indices.into_boxed_slice();
        self.palette_mode = true;
        self.dirty = true;
    }

    /// Restore from flat block data (for load).
    pub fn restore_flat(&mut self, blocks: Vec<BlockId>) {
        self.flat = blocks.into_boxed_slice();
        self.palette.clear();
        self.palette_mode = false;
        self.dirty = true;
    }

    /// Restore sunlight data (for load).
    pub fn restore_sunlight(&mut self, data: Vec<u8>) {
        self.sunlight = data.into_boxed_slice();
        self.light_dirty = true;
        self.dirty = true;
    }

    /// Restore torchlight data (for load).
    pub fn restore_torchlight(&mut self, data: Vec<u8>) {
        self.torchlight = data.into_boxed_slice();
        self.light_dirty = true;
        self.dirty = true;
    }

    /// Restore water level data (for load).
    pub fn restore_water_level(&mut self, data: Vec<u8>) {
        self.water_level = data.into_boxed_slice();
        self.light_dirty = true;
        self.dirty = true;
    }
}

#[inline]
fn in_bounds(x: i32, y: i32, z: i32) -> bool {
    (0..voxel_core::CHUNK_SIZE).contains(&x)
        && (0..voxel_core::CHUNK_SIZE).contains(&y)
        && (0..voxel_core::CHUNK_SIZE).contains(&z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxel_core::{ChunkPos, CHUNK_SIZE};

    fn test_chunk() -> Chunk {
        Chunk::new(ChunkPos::new(0, 0, 0))
    }

    #[test]
    fn new_chunk_is_all_air() {
        let c = test_chunk();
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert!(c.get(x, y, z).is_air());
                }
            }
        }
    }

    #[test]
    fn set_and_get() {
        let mut c = test_chunk();
        let stone = BlockId(2);
        c.set(3, 5, 7, stone);
        assert_eq!(c.get(3, 5, 7), stone);
    }

    #[test]
    fn out_of_bounds_returns_air() {
        let c = test_chunk();
        assert!(c.get(-1, 0, 0).is_air());
        assert!(c.get(CHUNK_SIZE, 0, 0).is_air());
        assert!(c.get(0, -1, 0).is_air());
        assert!(c.get(0, 0, CHUNK_SIZE).is_air());
    }

    #[test]
    fn non_air_count() {
        let mut c = test_chunk();
        assert_eq!(c.non_air_count(), 0);
        c.set(0, 0, 0, BlockId(1));
        c.set(1, 1, 1, BlockId(2));
        assert_eq!(c.non_air_count(), 2);
    }

    #[test]
    fn fill_uniform() {
        let mut c = test_chunk();
        c.fill_uniform(BlockId(5));
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(c.get(x, y, z), BlockId(5));
                }
            }
        }
    }

    #[test]
    fn fill_with() {
        let mut c = test_chunk();
        c.fill_with(|x, y, z| BlockId((x + y + z) as u16));
        assert_eq!(c.get(0, 0, 0), BlockId(0));
        assert_eq!(c.get(1, 2, 3), BlockId(6));
    }

    #[test]
    fn sunlight_defaults() {
        let c = test_chunk();
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(c.get_sunlight(x, y, z), 0);
                }
            }
        }
    }

    #[test]
    fn set_get_sunlight() {
        let mut c = test_chunk();
        c.set_sunlight(3, 5, 7, 15);
        assert_eq!(c.get_sunlight(3, 5, 7), 15);
    }

    #[test]
    fn torchlight_defaults() {
        let c = test_chunk();
        assert_eq!(c.get_torchlight(0, 0, 0), 0);
    }

    #[test]
    fn set_get_torchlight() {
        let mut c = test_chunk();
        c.set_torchlight(2, 3, 4, 12);
        assert_eq!(c.get_torchlight(2, 3, 4), 12);
    }

    #[test]
    fn clear_light() {
        let mut c = test_chunk();
        c.set_sunlight(0, 0, 0, 15);
        c.set_torchlight(1, 1, 1, 10);
        c.clear_light();
        assert_eq!(c.get_sunlight(0, 0, 0), 0);
        assert_eq!(c.get_torchlight(1, 1, 1), 0);
    }

    #[test]
    fn column_height_empty() {
        let c = test_chunk();
        assert_eq!(c.column_height(0, 0), 0);
    }

    #[test]
    fn column_height_with_block() {
        let mut c = test_chunk();
        c.set(0, 10, 0, BlockId(1));
        assert_eq!(c.column_height(0, 0), 11); // y + 1
    }

    #[test]
    fn solid_count_empty() {
        let c = test_chunk();
        let reg = crate::registry::BlockRegistry::with_builtins();
        assert_eq!(c.solid_count(&reg), 0);
    }

    // --- Palette mode tests ---

    #[test]
    fn palette_mode_activates_with_few_unique_blocks() {
        let mut c = test_chunk();
        // Set several blocks of the same type.
        for i in 0..20 {
            c.set(i, 0, 0, BlockId(1));
        }
        // 2 unique blocks (air + BlockId(1)) is well under 128, so palette mode activates.
        assert!(c.is_palette_mode());
    }

    #[test]
    fn fill_uniform_uses_palette() {
        let mut c = test_chunk();
        c.fill_uniform(BlockId(5));
        // Single block type should use palette mode.
        assert!(c.is_palette_mode());
        assert_eq!(c.palette_len(), 1);
        assert_eq!(c.get(0, 0, 0), BlockId(5));
        assert_eq!(c.get(15, 15, 15), BlockId(5));
    }

    #[test]
    fn palette_get_set_roundtrip() {
        let mut c = test_chunk();
        c.fill_uniform(BlockId(1));
        // Now set a different block.
        c.set(5, 5, 5, BlockId(2));
        assert_eq!(c.get(5, 5, 5), BlockId(2));
        assert_eq!(c.get(4, 4, 4), BlockId(1));
        // Palette should have grown.
        assert!(c.palette_len() >= 2);
    }

    // --- water level tests ---

    #[test]
    fn water_level_defaults_to_zero() {
        let c = test_chunk();
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(c.get_water_level(x, y, z), 0);
                }
            }
        }
    }

    #[test]
    fn set_get_water_level() {
        let mut c = test_chunk();
        c.set_water_level(3, 5, 7, 8);
        assert_eq!(c.get_water_level(3, 5, 7), 8);
    }

    #[test]
    fn water_level_out_of_bounds_returns_zero() {
        let c = test_chunk();
        assert_eq!(c.get_water_level(-1, 0, 0), 0);
        assert_eq!(c.get_water_level(CHUNK_SIZE, 0, 0), 0);
        assert_eq!(c.get_water_level(0, -1, 0), 0);
        assert_eq!(c.get_water_level(0, 0, CHUNK_SIZE), 0);
    }

    #[test]
    fn water_level_set_out_of_bounds_is_noop() {
        let mut c = test_chunk();
        c.set_water_level(-1, 0, 0, 8);
        c.set_water_level(CHUNK_SIZE, 0, 0, 8);
        assert_eq!(c.get_water_level(0, 0, 0), 0);
    }

    #[test]
    fn water_level_all_eight_values() {
        let mut c = test_chunk();
        for level in 0u8..=8 {
            c.set_water_level(0, 0, 0, level);
            assert_eq!(c.get_water_level(0, 0, 0), level);
        }
    }

    #[test]
    fn water_level_dirty_on_change() {
        let mut c = test_chunk();
        c.dirty = false;
        c.set_water_level(0, 0, 0, 5);
        assert!(c.dirty);
    }

    #[test]
    fn water_level_no_dirty_on_same_value() {
        let mut c = test_chunk();
        c.set_water_level(0, 0, 0, 5);
        c.dirty = false;
        c.set_water_level(0, 0, 0, 5);
        assert!(!c.dirty);
    }

    #[test]
    fn water_level_multiple_positions() {
        let mut c = test_chunk();
        c.set_water_level(0, 0, 0, 1);
        c.set_water_level(5, 5, 5, 8);
        c.set_water_level(15, 15, 15, 4);
        assert_eq!(c.get_water_level(0, 0, 0), 1);
        assert_eq!(c.get_water_level(5, 5, 5), 8);
        assert_eq!(c.get_water_level(15, 15, 15), 4);
        // Other positions remain zero.
        assert_eq!(c.get_water_level(1, 0, 0), 0);
        assert_eq!(c.get_water_level(5, 5, 6), 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn get_set_roundtrip(x in 0i32..16, y in 0i32..16, z in 0i32..16, id in 1u16..255) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            let block = BlockId(id);
            c.set(x, y, z, block);
            prop_assert_eq!(c.get(x, y, z), block);
        }

        #[test]
        fn out_of_bounds_get_is_air(x in -100i32..0, y in 0i32..16, z in 0i32..16) {
            let c = Chunk::new(ChunkPos::new(0, 0, 0));
            prop_assert_eq!(c.get(x, y, z), BlockId::AIR);
        }

        #[test]
        fn out_of_bounds_set_is_noop(x in -100i32..0, y in 0i32..16, z in 0i32..16) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set(x, y, z, BlockId(5));
            prop_assert!(c.get(0, 0, 0).is_air());
        }

        #[test]
        fn set_same_value_no_dirty(x in 0i32..16, y in 0i32..16, z in 0i32..16) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set(x, y, z, BlockId::AIR); // already air
            prop_assert!(!c.dirty);
        }

        #[test]
        fn sunlight_get_set(x in 0i32..16, y in 0i32..16, z in 0i32..16, lvl in 0u8..16) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set_sunlight(x, y, z, lvl);
            prop_assert_eq!(c.get_sunlight(x, y, z), lvl);
        }

        #[test]
        fn torchlight_get_set(x in 0i32..16, y in 0i32..16, z in 0i32..16, lvl in 0u8..16) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set_torchlight(x, y, z, lvl);
            prop_assert_eq!(c.get_torchlight(x, y, z), lvl);
        }

        #[test]
        fn water_level_get_set(x in 0i32..16, y in 0i32..16, z in 0i32..16, lvl in 0u8..9) {
            let mut c = Chunk::new(ChunkPos::new(0, 0, 0));
            c.set_water_level(x, y, z, lvl);
            prop_assert_eq!(c.get_water_level(x, y, z), lvl);
        }
    }
}
