//! Procedural terrain generation.
//!
//! Pipeline per chunk column (x,z):
//!   1. Biome is chosen from a temperature/humidity noise field.
//!   2. A continent/height field sets the base elevation; biome modifies it
//!      (mountains taller, oceans lower, etc.).
//!   3. Surface depth selects grass/dirt/stone/sand/snow by height + biome.
//!   4. Caves carve 3D noise tunnels; ores sprinkle by depth.
//!   5. Trees and small decorations scatter on grass surfaces.
//!
//! Generation is deterministic from a world seed and entirely `rayon`-parallel
//! at the chunk level (see `ChunkStreamer`).

use fastnoise_lite::FastNoiseLite;
use voxel_core::{
    math::{chunk_origin, ChunkPos},
    BlockId, BlockPos, CHUNK_SIZE, SEA_LEVEL, WORLD_HEIGHT_BLOCKS,
};

use crate::{chunk::Chunk, registry::BlockRegistry};

/// Identifier for a surface biome.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiomeId {
    Ocean,
    Plains,
    Forest,
    Desert,
    Mountains,
    Beach,
}

impl BiomeId {
    /// Pick a biome from temperature (0..1) and humidity (0..1) plus height.
    fn classify(temp: f32, humid: f32, height: f32) -> Self {
        if height < (SEA_LEVEL - 8) as f32 {
            BiomeId::Ocean
        } else if height < (SEA_LEVEL + 1) as f32 {
            if temp > 0.6 {
                BiomeId::Beach
            } else {
                BiomeId::Plains
            }
        } else if height > (SEA_LEVEL + 40) as f32 {
            BiomeId::Mountains
        } else if temp > 0.7 {
            BiomeId::Desert
        } else if humid > 0.55 {
            BiomeId::Forest
        } else {
            BiomeId::Plains
        }
    }
}

/// Configurable terrain generator. Owns precomputed noise samplers.
pub struct TerrainGenerator {
    seed: i32,
    // Continental shape (low frequency).
    continent: FastNoiseLite,
    // Hills / rolling terrain (medium frequency).
    hills: FastNoiseLite,
    // Mountain ridges (ridge noise).
    ridge: FastNoiseLite,
    // Biome temperature.
    temperature: FastNoiseLite,
    // Biome humidity.
    humidity: FastNoiseLite,
    // 3D cave noise.
    cave: FastNoiseLite,
    // Ore placement jitter.
    ore: FastNoiseLite,
    // Tree scatter.
    tree: FastNoiseLite,
}

impl TerrainGenerator {
    pub fn new(seed: i32) -> Self {
        let make = |freq: f32, ty: fastnoise_lite::NoiseType| {
            let mut n = FastNoiseLite::new();
            n.set_seed(Some(seed));
            n.set_noise_type(Some(ty));
            n.set_frequency(Some(freq));
            n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
            n.set_fractal_octaves(Some(4));
            n.set_fractal_lacunarity(Some(2.0));
            n.set_fractal_gain(Some(0.5));
            n
        };
        Self {
            seed,
            continent: make(0.0018, fastnoise_lite::NoiseType::OpenSimplex2),
            hills: make(0.01, fastnoise_lite::NoiseType::OpenSimplex2),
            ridge: make(0.0035, fastnoise_lite::NoiseType::OpenSimplex2),
            temperature: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(3)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.0015));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            humidity: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(7)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.0015));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            cave: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(11)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::OpenSimplex2));
                n.set_frequency(Some(0.015));
                n.set_fractal_type(Some(fastnoise_lite::FractalType::FBm));
                n.set_fractal_octaves(Some(3));
                n
            },
            ore: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(13)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
                n.set_frequency(Some(0.06));
                n
            },
            tree: {
                let mut n = FastNoiseLite::new();
                n.set_seed(Some(seed.wrapping_mul(17)));
                n.set_noise_type(Some(fastnoise_lite::NoiseType::Cellular));
                n.set_frequency(Some(0.12));
                n
            },
        }
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }

    /// Base terrain height (in world Y block units) for a world (x, z).
    fn base_height(&self, x: i32, z: i32) -> f32 {
        let xf = x as f32;
        let zf = z as f32;
        let cont = self.continent.get_noise_2d(xf, zf); // -1..1
        let hills = self.hills.get_noise_2d(xf, zf);
        let ridge = 1.0 - (self.ridge.get_noise_2d(xf, zf).abs()); // 0..1, ridged

        // Continental: large sea-level baseline ± some land/ocean.
        let base = SEA_LEVEL as f32 + cont * 24.0;
        // Hills add rolling variation on land.
        let hill_part = hills * 8.0;
        // Mountains: only where continental is high (land interior).
        let mountain_mask = ((cont - 0.2) / 0.8).clamp(0.0, 1.0);
        let mountain_part = ridge * ridge * 60.0 * mountain_mask;

        base + hill_part + mountain_part
    }

    /// Search outward from the world origin for a land column above sea level.
    /// Returns (x, surface_height, z) so the caller can place a spawn there.
    /// Uses the height function directly — no chunk loading required.
    pub fn find_spawn(&self) -> (i32, i32, i32) {
        for r in 0..128i32 {
            if r == 0 {
                let h = self.base_height(0, 0).round() as i32;
                if h > SEA_LEVEL + 2 {
                    return (0, h, 0);
                }
                continue;
            }
            // Walk the ring at radius r.
            for dx in -r..=r {
                for &dz in &[-r, r] {
                    let h = self.base_height(dx, dz).round() as i32;
                    if h > SEA_LEVEL + 2 {
                        return (dx, h, dz);
                    }
                }
            }
            for &dx in &[-r, r] {
                for dz in (-r + 1)..r {
                    let h = self.base_height(dx, dz).round() as i32;
                    if h > SEA_LEVEL + 2 {
                        return (dx, h, dz);
                    }
                }
            }
        }
        (0, 90, 0) // fallback: high in the air at origin
    }

    fn biome_at(&self, x: i32, z: i32, height: f32) -> BiomeId {
        let t = (self.temperature.get_noise_2d(x as f32, z as f32) + 1.0) * 0.5;
        let h = (self.humidity.get_noise_2d(x as f32, z as f32) + 1.0) * 0.5;
        BiomeId::classify(t, h, height)
    }

    /// Is the block at (world_x, world_y, world_z) carved out by a cave?
    fn is_cave(&self, x: i32, y: i32, z: i32) -> bool {
        if !(4..=WORLD_HEIGHT_BLOCKS - 8).contains(&y) {
            return false;
        }
        let n = self.cave.get_noise_3d(x as f32, y as f32 * 1.5, z as f32);
        // Two overlapping noise thresholds create winding tunnels.
        n.abs() < 0.06
    }

    /// Pick an ore block for a stone block at depth `y`, or None for plain stone.
    /// `noise_val` is the pre-computed ore noise value (-1..1).
    fn ore_for_val(&self, noise_val: f32, y: i32, reg: &BlockRegistry) -> Option<BlockId> {
        let v = (noise_val + 1.0) * 0.5; // 0..1
        if y < 16 && v > 0.985 {
            reg.id_of("diamond_ore")
        } else if y < 32 && v > 0.97 {
            reg.id_of("gold_ore")
        } else if y < 64 && v > 0.92 {
            reg.id_of("iron_ore")
        } else if v > 0.86 {
            reg.id_of("coal_ore")
        } else {
            None
        }
    }

    /// Generate a full chunk in place. Does NOT touch neighbours; cross-chunk
    /// decorations (trees) are applied by `decorate` after neighbours exist.
    pub fn generate(&self, chunk: &mut Chunk, reg: &BlockRegistry) {
        let origin = chunk_origin(chunk.pos);
        let stone = reg.id_of("stone").unwrap();
        let dirt = reg.id_of("dirt").unwrap();
        let grass = reg.id_of("grass").unwrap();
        let sand = reg.id_of("sand").unwrap();
        let water = reg.id_of("water").unwrap();
        let bedrock = reg.id_of("bedrock").unwrap();
        let snow = reg.id_of("snow").unwrap();
        let gravel = reg.id_of("gravel").unwrap();

        for lx in 0..CHUNK_SIZE {
            for lz in 0..CHUNK_SIZE {
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                let height_f = self.base_height(wx, wz);
                let height = height_f.round() as i32;
                let biome = self.biome_at(wx, wz, height_f);

                for ly in 0..CHUNK_SIZE {
                    let wy = origin.y + ly;
                    if wy >= WORLD_HEIGHT_BLOCKS {
                        chunk.set(lx, ly, lz, BlockId::AIR);
                        continue;
                    }

                    // Bedrock floor at the very bottom of the world.
                    let mut block = if wy == 0 {
                        bedrock
                    } else if wy < height - 4 {
                        // Deep: stone with possible ores, or gravel pockets.
                        let deep = self.ore.get_noise_3d(wx as f32, wy as f32, wz as f32);
                        let deep_mapped = deep * 2.0 - 1.0;
                        if deep > 0.78 {
                            gravel
                        } else {
                            self.ore_for_val(deep_mapped, wy, reg).unwrap_or(stone)
                        }
                    } else if wy < height - 1 {
                        // Subsurface: dirt (or sand in desert/beach).
                        match biome {
                            BiomeId::Desert | BiomeId::Beach => sand,
                            _ => dirt,
                        }
                    } else if wy < height {
                        // Surface block.
                        match biome {
                            BiomeId::Desert | BiomeId::Beach => sand,
                            BiomeId::Mountains if wy > SEA_LEVEL + 55 => snow,
                            BiomeId::Ocean if wy < SEA_LEVEL => gravel,
                            // Underwater surfaces get dirt, not grass.
                            _ if height <= SEA_LEVEL => dirt,
                            _ => grass,
                        }
                    } else if wy <= SEA_LEVEL {
                        // Below sea level and above terrain: water (oceans/lakes).
                        water
                    } else {
                        BlockId::AIR
                    };

                    // Carve caves — but keep the top crust intact so caves
                    // never break through the surface or puncture the ocean
                    // floor. This prevents seeing caves through water and
                    // avoids surface potholes. Bedrock and water are never carved.
                    let crust_bottom = height - 3;
                    if block != bedrock
                        && block != water
                        && wy < crust_bottom
                        && self.is_cave(wx, wy, wz)
                    {
                        block = BlockId::AIR;
                    }

                    chunk.set(lx, ly, lz, block);
                    // Water placed during worldgen is always a full source block.
                    if block == water {
                        chunk.set_water_level(lx, ly, lz, 8);
                    }
                }
            }
        }
        chunk.generated = true;
        chunk.dirty = true;
    }

    /// Scatter trees on grass surfaces. Call after the chunk and its neighbours
    /// are generated so trunks/leaves can spill across borders safely.
    pub fn decorate(
        &self,
        chunk: &mut Chunk,
        reg: &BlockRegistry,
        neighbour_sample: impl Fn(i32, i32, i32) -> BlockId,
    ) {
        let origin = chunk_origin(chunk.pos);
        let grass = match reg.id_of("grass") {
            Some(g) => g,
            None => return,
        };
        let wood = match reg.id_of("wood") {
            Some(w) => w,
            None => return,
        };
        let leaves = match reg.id_of("leaves") {
            Some(l) => l,
            None => return,
        };
        // New decorative blocks. None of these abort decoration: a missing block
        // just skips that particular feature so the rest can still be placed.
        let birch_log = reg.id_of("birch_log");
        let birch_leaves = reg.id_of("birch_leaves");
        let spruce_log = reg.id_of("spruce_log");
        let spruce_leaves = reg.id_of("spruce_leaves");
        let tall_grass = reg.id_of("tall_grass");
        let poppy = reg.id_of("poppy");
        let dandelion = reg.id_of("dandelion");
        let cactus = reg.id_of("cactus");
        let mushroom_red = reg.id_of("mushroom_red");
        let mushroom_brown = reg.id_of("mushroom_brown");
        let sand = reg.id_of("sand");

        // Which tree variety to plant for a given column.
        #[derive(Clone, Copy)]
        enum TreeType {
            Oak,
            Birch,
            Spruce,
        }

        // Deterministic per-column tree decision from cellular noise + hash.
        for lx in 2..CHUNK_SIZE - 2 {
            for lz in 2..CHUNK_SIZE - 2 {
                let wx = origin.x + lx;
                let wz = origin.z + lz;
                // Tree density: cellular value high + per-position hash.
                let n = self.tree.get_noise_2d(wx as f32, wz as f32);
                let h = hash2(self.seed, wx, wz);
                // Biome is derived from the base height field (same source as
                // `generate`), so decorations match the surface biome exactly.
                let height_f = self.base_height(wx, wz);
                let biome = self.biome_at(wx, wz, height_f);

                // ---- Trees ------------------------------------------------
                if n >= 0.55 && h <= 0.12 {
                    // Find a grass surface by scanning down from chunk top.
                    let mut surface_y = None;
                    for ly in (0..CHUNK_SIZE).rev() {
                        let wy = origin.y + ly;
                        if wy >= WORLD_HEIGHT_BLOCKS {
                            continue;
                        }
                        let b = chunk.get(lx, ly, lz);
                        if b == grass {
                            surface_y = Some(ly);
                            break;
                        }
                        if !b.is_air() {
                            break; // hit non-grass solid before grass -> no tree
                        }
                    }
                    if let Some(sy) = surface_y {
                        // Pick tree variety by biome + secondary hash. Oak is
                        // the default/fallback when no biome variant fires.
                        let tree_hash = hash2(self.seed, wx + 1, wz);
                        let tree_type = match biome {
                            BiomeId::Forest if tree_hash < 0.3 => TreeType::Birch,
                            BiomeId::Mountains if tree_hash < 0.4 => TreeType::Spruce,
                            _ => TreeType::Oak,
                        };
                        match tree_type {
                            TreeType::Oak => {
                                let trunk_top = (sy + 5).min(CHUNK_SIZE - 1);
                                for ty in (sy + 1)..=trunk_top {
                                    chunk.set(lx, ty, lz, wood);
                                }
                                // Leaf canopy: a 3×3×2 blob centred on trunk_top.
                                let cy = trunk_top;
                                for dy in 0..=2 {
                                    for dx in -2i32..=2 {
                                        for dz in -2i32..=2 {
                                            if dx == 0 && dz == 0 && dy < 2 {
                                                continue; // leave the trunk
                                            }
                                            if dx.abs() == 2 && dz.abs() == 2 {
                                                continue; // round corners
                                            }
                                            let lx2 = lx + dx;
                                            let lz2 = lz + dz;
                                            let ly2 = cy + dy;
                                            if (0..CHUNK_SIZE).contains(&lx2)
                                                && (0..CHUNK_SIZE).contains(&lz2)
                                                && (0..CHUNK_SIZE).contains(&ly2)
                                                && chunk.get(lx2, ly2, lz2).is_air()
                                            {
                                                chunk.set(lx2, ly2, lz2, leaves);
                                            }
                                        }
                                    }
                                }
                            }
                            TreeType::Birch => {
                                if let (Some(log), Some(leaf)) = (birch_log, birch_leaves) {
                                    // 6-7 tall trunk.
                                    let trunk_h =
                                        6 + (hash2(self.seed, wx + 3, wz + 3) * 2.0) as i32;
                                    let trunk_top = (sy + trunk_h).min(CHUNK_SIZE - 1);
                                    for ty in (sy + 1)..=trunk_top {
                                        chunk.set(lx, ty, lz, log);
                                    }
                                    // Small 2×2×2 leaf canopy at the top.
                                    for dy in 0..2 {
                                        let ly2 = trunk_top + dy;
                                        for dx in -1i32..=0 {
                                            for dz in -1i32..=0 {
                                                let lx2 = lx + dx;
                                                let lz2 = lz + dz;
                                                if (0..CHUNK_SIZE).contains(&lx2)
                                                    && (0..CHUNK_SIZE).contains(&lz2)
                                                    && (0..CHUNK_SIZE).contains(&ly2)
                                                    && chunk.get(lx2, ly2, lz2).is_air()
                                                {
                                                    chunk.set(lx2, ly2, lz2, leaf);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            TreeType::Spruce => {
                                if let (Some(log), Some(leaf)) = (spruce_log, spruce_leaves) {
                                    // 6-8 tall trunk.
                                    let trunk_h =
                                        6 + (hash2(self.seed, wx + 5, wz + 5) * 3.0) as i32;
                                    let trunk_top = (sy + trunk_h).min(CHUNK_SIZE - 1);
                                    for ty in (sy + 1)..=trunk_top {
                                        chunk.set(lx, ty, lz, log);
                                    }
                                    // Narrow tapered canopy: 3×3 at the bottom
                                    // narrowing to 1×1 at the very top.
                                    let canopy_top = trunk_top;
                                    let canopy_bottom = trunk_top - 2;
                                    for ly2 in canopy_bottom..=canopy_top {
                                        let dist_from_top = canopy_top - ly2;
                                        let radius = if dist_from_top == 0 { 0 } else { 1 };
                                        for dx in -radius..=radius {
                                            for dz in -radius..=radius {
                                                // Leave the trunk except at the top.
                                                if dx == 0
                                                    && dz == 0
                                                    && ly2 < canopy_top
                                                {
                                                    continue;
                                                }
                                                let lx2 = lx + dx;
                                                let lz2 = lz + dz;
                                                if (0..CHUNK_SIZE).contains(&lx2)
                                                    && (0..CHUNK_SIZE).contains(&lz2)
                                                    && (0..CHUNK_SIZE).contains(&ly2)
                                                    && chunk.get(lx2, ly2, lz2).is_air()
                                                {
                                                    chunk.set(lx2, ly2, lz2, leaf);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                // ---- Surface foliage scatter ------------------------------
                // Find the surface block (grass, or sand in desert/beach) by
                // scanning down. Trees above would have caused an early break
                // here, so foliage naturally avoids tree trunks.
                let mut foliage_surface = None;
                for ly in (0..CHUNK_SIZE).rev() {
                    let wy = origin.y + ly;
                    if wy >= WORLD_HEIGHT_BLOCKS {
                        continue;
                    }
                    let b = chunk.get(lx, ly, lz);
                    let is_grass = b == grass;
                    let is_sand = sand.map(|s| b == s).unwrap_or(false);
                    if is_grass || is_sand {
                        foliage_surface = Some(ly);
                        break;
                    }
                    if !b.is_air() {
                        break;
                    }
                }
                if let Some(sy) = foliage_surface {
                    let fy = sy + 1;
                    if fy < CHUNK_SIZE && chunk.get(lx, fy, lz).is_air() {
                        match biome {
                            BiomeId::Plains => {
                                if h < 0.02 {
                                    // Rare flower: poppy or dandelion.
                                    if hash2(self.seed, wx, wz + 100) < 0.5 {
                                        if let Some(p) = poppy {
                                            chunk.set(lx, fy, lz, p);
                                        }
                                    } else if let Some(d) = dandelion {
                                        chunk.set(lx, fy, lz, d);
                                    }
                                } else if h < 0.15 {
                                    if let Some(tg) = tall_grass {
                                        chunk.set(lx, fy, lz, tg);
                                    }
                                }
                            }
                            BiomeId::Forest => {
                                if h < 0.03 {
                                    // Mushroom: red or brown.
                                    if hash2(self.seed, wx, wz + 100) < 0.5 {
                                        if let Some(m) = mushroom_red {
                                            chunk.set(lx, fy, lz, m);
                                        }
                                    } else if let Some(m) = mushroom_brown {
                                        chunk.set(lx, fy, lz, m);
                                    }
                                } else if h < 0.10 {
                                    if let Some(tg) = tall_grass {
                                        chunk.set(lx, fy, lz, tg);
                                    }
                                }
                            }
                            BiomeId::Desert => {
                                if h < 0.05 {
                                    if let Some(c) = cactus {
                                        let cactus_h = 1
                                            + (hash2(self.seed, wx + 7, wz + 7) * 3.0) as i32;
                                        for i in 0..cactus_h {
                                            let cy = sy + 1 + i;
                                            if cy < CHUNK_SIZE
                                                && chunk.get(lx, cy, lz).is_air()
                                            {
                                                chunk.set(lx, cy, lz, c);
                                            }
                                        }
                                    }
                                }
                            }
                            BiomeId::Mountains => {
                                if h < 0.08 {
                                    if let Some(tg) = tall_grass {
                                        chunk.set(lx, fy, lz, tg);
                                    }
                                }
                            }
                            BiomeId::Beach | BiomeId::Ocean => {}
                        }
                    }
                }
            }
        }
        // neighbour_sample is reserved for future cross-chunk foliage; keep the
        // parameter so callers don't need to change when we add it.
        let _ = neighbour_sample;
    }

    /// Place dungeons, ruined towers and wells. Each chunk independently
    /// recomputes which structures overlap it (deterministically, via the
    /// world-seed hash at the structure's anchor column) and writes only the
    /// blocks that fall inside its own bounds; `chunk.set` no-ops on
    /// out-of-range locals, so cross-chunk structures are stitched together
    /// by each chunk placing its own slice. `sample` reads world blocks
    /// across chunk borders for placement-condition checks.
    pub fn place_structures(
        &self,
        chunk: &mut Chunk,
        reg: &BlockRegistry,
        sample: &dyn Fn(i32, i32, i32) -> voxel_core::BlockId,
    ) {
        let origin = chunk_origin(chunk.pos);
        let (
            Some(stone),
            Some(mossy),
            Some(chest),
            Some(cobble),
            Some(grass),
            Some(dirt),
            Some(water),
        ) = (
            reg.id_of("stone"),
            reg.id_of("mossy_cobblestone"),
            reg.id_of("chest"),
            reg.id_of("cobblestone"),
            reg.id_of("grass"),
            reg.id_of("dirt"),
            reg.id_of("water"),
        )
        else {
            return;
        };

        // Dungeons: 5×3×5 underground rooms, anchor at (ax, ay, az).
        const DS: i32 = 5;
        const DH: i32 = 3;
        for ax in (origin.x - (DS - 1))..=(origin.x + CHUNK_SIZE - 1) {
            for az in (origin.z - (DS - 1))..=(origin.z + CHUNK_SIZE - 1) {
                if hash2(self.seed, ax, az) >= 0.003 {
                    continue;
                }
                let wy = 10 + (hash2(self.seed.wrapping_mul(2), ax, az) * 30.0) as i32;
                if !(10..=40).contains(&wy) {
                    continue;
                }
                // Only carve into solid stone (we're underground): check the
                // block just above the ceiling.
                if sample(ax + 2, wy + DH, az + 2) != stone {
                    continue;
                }
                // Don't spawn in or near water.
                if sample(ax + 2, wy, az + 2) == water {
                    continue;
                }
                for dx in 0..DS {
                    for dy in 0..DH {
                        for dz in 0..DS {
                            let on_shell = dx == 0
                                || dx == DS - 1
                                || dz == 0
                                || dz == DS - 1
                                || dy == 0
                                || dy == DH - 1;
                            let id = if on_shell { mossy } else { BlockId::AIR };
                            chunk.set(
                                ax + dx - origin.x,
                                wy + dy - origin.y,
                                az + dz - origin.z,
                                id,
                            );
                        }
                    }
                }
                // Chest on the centre floor.
                chunk.set(ax + 2 - origin.x, wy - origin.y, az + 2 - origin.z, chest);
            }
        }

        // Ruined towers: 3×3 hollow cobblestone shell on the surface.
        const TS: i32 = 3;
        for ax in (origin.x - (TS - 1))..=(origin.x + CHUNK_SIZE - 1) {
            for az in (origin.z - (TS - 1))..=(origin.z + CHUNK_SIZE - 1) {
                if hash2(self.seed, ax, az) >= 0.001 {
                    continue;
                }
                let surf_f = self.base_height(ax, az);
                let surf = surf_f.round() as i32;
                let surface_block = sample(ax, surf - 1, az);
                if surface_block != grass && surface_block != dirt {
                    continue;
                }
                // Don't spawn towers underwater.
                if surf <= SEA_LEVEL {
                    continue;
                }
                let height = 6 + (hash2(self.seed.wrapping_mul(3), ax, az) * 5.0) as i32;
                for dx in 0..TS {
                    for dz in 0..TS {
                        let is_wall = dx == 0 || dx == TS - 1 || dz == 0 || dz == TS - 1;
                        if !is_wall {
                            continue;
                        }
                        for dy in 0..height {
                            // 1-block door gap at ground level on the +X face.
                            if dy == 0 && dx == TS - 1 && dz == 1 {
                                continue;
                            }
                            chunk.set(
                                ax + dx - origin.x,
                                surf + dy - origin.y,
                                az + dz - origin.z,
                                cobble,
                            );
                        }
                    }
                }
            }
        }

        // Wells: 3×3×4 cobblestone ring (dry), Desert/Plains only.
        const WS: i32 = 3;
        const WH: i32 = 4;
        for ax in (origin.x - (WS - 1))..=(origin.x + CHUNK_SIZE - 1) {
            for az in (origin.z - (WS - 1))..=(origin.z + CHUNK_SIZE - 1) {
                if hash2(self.seed, ax, az) >= 0.005 {
                    continue;
                }
                let surf_f = self.base_height(ax, az);
                let surf = surf_f.round() as i32;
                let biome = self.biome_at(ax, az, surf_f);
                if biome != BiomeId::Desert && biome != BiomeId::Plains {
                    continue;
                }
                // Don't spawn wells underwater.
                if surf <= SEA_LEVEL {
                    continue;
                }
                for dx in 0..WS {
                    for dz in 0..WS {
                        let is_center = dx == 1 && dz == 1;
                        let id = if is_center { BlockId::AIR } else { cobble };
                        for dy in 0..WH {
                            chunk.set(
                                ax + dx - origin.x,
                                surf + dy - origin.y,
                                az + dz - origin.z,
                                id,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Height of the highest non-air block in world column (x, z), or 0.
    pub fn column_height(&self, chunk: &Chunk) -> i32 {
        // Scan columns for the highest non-air block.
        let origin = chunk_origin(chunk.pos);
        let mut max_height = -1i32;
        for lx in 0..voxel_core::CHUNK_SIZE {
            for lz in 0..voxel_core::CHUNK_SIZE {
                let h = chunk.column_height(lx, lz);
                if h > max_height {
                    max_height = h;
                }
            }
        }
        origin.y + max_height + 1
    }
}

/// Deterministic hash of (seed, x, z) into [0, 1).
fn hash2(seed: i32, x: i32, z: i32) -> f32 {
    let mut h = (seed as u32).wrapping_mul(374761393);
    h = h.wrapping_add(x as u32).wrapping_mul(668265263);
    h = h.wrapping_add(z as u32).wrapping_mul(1274126177);
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    (h >> 8) as f32 / ((1u32 << 24) as f32)
}

/// Convert a world block position to its owning chunk position.
pub fn chunk_of(block: BlockPos) -> ChunkPos {
    voxel_core::math::block_to_chunk(block.0)
}
