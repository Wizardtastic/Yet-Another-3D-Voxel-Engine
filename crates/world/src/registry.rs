//! Block registry: maps `BlockId` -> properties (solidity, opacity, textures).
//!
//! This is the runtime side of the data-driven content system. In a future
//! iteration `voxel-assets` populates this from JSON; for now we build a
//! hardcoded builtin set so the world has something to generate with.

use std::sync::Arc;

use glam::IVec3;
use voxel_core::BlockId;

/// The six cube faces, in the order the mesher and shaders expect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Face {
    NegX = 0,
    PosX = 1,
    NegY = 2,
    PosY = 3,
    NegZ = 4,
    PosZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::NegX,
        Face::PosX,
        Face::NegY,
        Face::PosY,
        Face::NegZ,
        Face::PosZ,
    ];

    /// Outward unit normal for this face.
    pub fn normal(self) -> IVec3 {
        match self {
            Face::NegX => IVec3::new(-1, 0, 0),
            Face::PosX => IVec3::new(1, 0, 0),
            Face::NegY => IVec3::new(0, -1, 0),
            Face::PosY => IVec3::new(0, 1, 0),
            Face::NegZ => IVec3::new(0, 0, -1),
            Face::PosZ => IVec3::new(0, 0, 1),
        }
    }
}

/// Coarse classification used by worldgen and physics. More kinds can be added
/// without breaking the registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum BlockKind {
    #[default]
    Air,
    Solid,
    Liquid,
    Foliage,
    Transparent,
}

/// Per-face texture tile index into the texture atlas.
#[derive(Clone, Copy, Debug)]
pub struct BlockTextures {
    /// One tile index per face; uses a single tile for all faces if `same` was
    /// used at registration.
    pub tiles: [u16; 6],
}

impl BlockTextures {
    /// All six faces share one tile (e.g. stone, planks).
    pub fn same(tile: u16) -> Self {
        Self { tiles: [tile; 6] }
    }
    /// Top/bottom/side variant (e.g. grass: grass_top, dirt, grass_side).
    pub fn top_bottom_side(top: u16, bottom: u16, side: u16) -> Self {
        // Face order: NegX, PosX, NegY, PosY, NegZ, PosZ
        Self {
            tiles: [side, side, bottom, top, side, side],
        }
    }
    pub fn tile(self, face: Face) -> u16 {
        self.tiles[face as usize]
    }
}

/// Static properties of a block type.
#[derive(Clone, Debug)]
pub struct BlockDef {
    pub id: BlockId,
    /// Owned name. `Arc<str>` so the registry can be cheaply cloned without
    /// aliasing a global / leaking the string; also makes runtime-loaded names
    /// from JSON safe.
    pub name: Arc<str>,
    pub kind: BlockKind,
    /// Whether the block blocks entity movement (solids + liquids do, air doesn't).
    pub solid: bool,
    /// Whether the block fully occludes neighbouring faces (air/glass/leaves don't).
    pub opaque: bool,
    /// Whether the block can be broken by the player. False for bedrock etc.
    pub breakable: bool,
    /// Whether a block here can be replaced by placement (air, tall grass, etc.).
    pub replaceable: bool,
    pub textures: BlockTextures,
    /// Light emitted by this block (0–15). 0 for most, 14 for torches.
    pub emission: u8,
    /// How much light this block absorbs (0–15). 0 = transparent, 15 = fully opaque.
    pub light_absorption: u8,
}

impl BlockDef {
    pub fn is_rendered(&self) -> bool {
        !matches!(self.kind, BlockKind::Air)
    }
}

/// Central block-definition table. Built once at startup; read by worldgen,
/// the mesher, and physics. Indexed by `BlockId`.
#[derive(Clone)]
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    /// Owned string keys so the map works for runtime-loaded names from JSON
    /// (we can't use `&'static str` for those, and we don't want a global arena
    /// to leak `Box<str>` to process lifetime).
    by_name: std::collections::HashMap<String, BlockId>,
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl BlockRegistry {
    /// Construct the registry populated with the builtin block set used by
    /// worldgen and the player. Tile indices match the atlas layout produced
    /// by `voxel_render::atlas`.
    pub fn with_builtins() -> Self {
        let mut reg = Self {
            defs: Vec::new(),
            by_name: std::collections::HashMap::new(),
        };
        // id 0 must be air.
        reg.add(BlockDef {
            id: BlockId::AIR,
            name: Arc::from("air"),
            kind: BlockKind::Air,
            solid: false,
            opaque: false,
            breakable: false,
            replaceable: true,
            textures: BlockTextures::same(0),
            emission: 0,
            light_absorption: 0,
        });
        // Atlas tile indices (see renderer atlas for the actual PNGs):
        // 0 air, 1 stone, 2 dirt, 3 grass_top, 4 grass_side, 5 sand,
        // 6 water, 7 wood_side, 8 wood_top, 9 leaves, 10 bedrock, 11 coal_ore,
        // 12 iron_ore, 13 gold_ore, 14 diamond_ore, 15 planks, 16 cobblestone,
        // 17 glass, 18 gravel, 19 snow, 20 white, 21 torch, 22 bucket,
        // 23 water_bucket, 24 tall_grass, 25 poppy, 26 dandelion, 27 cactus,
        // 28 mushroom_red, 29 mushroom_brown, 30 birch_log_side, 31 birch_log_top,
        // 32 birch_leaves, 33 spruce_log_side, 34 spruce_log_top, 35 spruce_leaves,
        // 36 mossy_cobblestone, 37 chest.
        reg.add_named("stone", solid_opaque(1));
        reg.add_named("dirt", solid_opaque(2));
        reg.add_named(
            "grass",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("grass"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(3, 2, 4),
                emission: 0,
                light_absorption: 15,
            },
        );
        reg.add_named("sand", solid_opaque(5));
        reg.add_named(
            "water",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("water"),
                kind: BlockKind::Liquid,
                solid: false,
                opaque: false,
                breakable: false,
                replaceable: true,
                textures: BlockTextures::same(6),
                emission: 0,
                light_absorption: 12,
            },
        );
        reg.add_named(
            "wood",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("wood"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(8, 8, 7),
                emission: 0,
                light_absorption: 15,
            },
        );
        reg.add_named(
            "leaves",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("leaves"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(9),
                emission: 0,
                light_absorption: 14,
            },
        );
        reg.add_named(
            "bedrock",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("bedrock"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: false,
                replaceable: false,
                textures: BlockTextures::same(10),
                emission: 0,
                light_absorption: 15,
            },
        );
        reg.add_named("coal_ore", solid_opaque(11));
        reg.add_named("iron_ore", solid_opaque(12));
        reg.add_named("gold_ore", solid_opaque(13));
        reg.add_named("diamond_ore", solid_opaque(14));
        reg.add_named("planks", solid_opaque(15));
        reg.add_named("cobblestone", solid_opaque(16));
        reg.add_named(
            "glass",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("glass"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(17),
                emission: 0,
                light_absorption: 14,
            },
        );
        reg.add_named("gravel", solid_opaque(18));
        reg.add_named("snow", solid_opaque(19));
        reg.add_named(
            "torch",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("torch"),
                kind: BlockKind::Transparent,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(21),
                emission: 14,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "bucket",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("bucket"),
                kind: BlockKind::Solid,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(22),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "water_bucket",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("water_bucket"),
                kind: BlockKind::Solid,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(23),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "tall_grass",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("tall_grass"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(24),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "poppy",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("poppy"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(25),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "dandelion",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("dandelion"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(26),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "cactus",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("cactus"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(38, 39, 27),
                emission: 0,
                light_absorption: 15,
            },
        );
        reg.add_named(
            "mushroom_red",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("mushroom_red"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(28),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "mushroom_brown",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("mushroom_brown"),
                kind: BlockKind::Foliage,
                solid: false,
                opaque: false,
                breakable: true,
                replaceable: true,
                textures: BlockTextures::same(29),
                emission: 0,
                light_absorption: 0,
            },
        );
        reg.add_named(
            "birch_log",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("birch_log"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(31, 31, 30),
                emission: 0,
                light_absorption: 15,
            },
        );
        reg.add_named(
            "birch_leaves",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("birch_leaves"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(32),
                emission: 0,
                light_absorption: 14,
            },
        );
        reg.add_named(
            "spruce_log",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("spruce_log"),
                kind: BlockKind::Solid,
                solid: true,
                opaque: true,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::top_bottom_side(34, 34, 33),
                emission: 0,
                light_absorption: 15,
            },
        );
        reg.add_named(
            "spruce_leaves",
            BlockDef {
                id: BlockId(0),
                name: Arc::from("spruce_leaves"),
                kind: BlockKind::Transparent,
                solid: true,
                opaque: false,
                breakable: true,
                replaceable: false,
                textures: BlockTextures::same(35),
                emission: 0,
                light_absorption: 14,
            },
        );
        reg.add_named("mossy_cobblestone", solid_opaque(36));
        reg.add_named("chest", solid_opaque(37));
        reg
    }

    /// Build a registry from asset-loaded block definitions.
    /// Air (id=0) is always prepended as the first block.
    pub fn from_assets(blocks: &[voxel_assets::BlockData]) -> Self {
        let mut reg = Self {
            defs: Vec::new(),
            by_name: std::collections::HashMap::new(),
        };
        // id 0 is always air (not in by_name).
        reg.add(BlockDef {
            id: BlockId::AIR,
            name: Arc::from("air"),
            kind: BlockKind::Air,
            solid: false,
            opaque: false,
            breakable: false,
            replaceable: true,
            textures: BlockTextures::same(0),
            emission: 0,
            light_absorption: 0,
        });
        for bd in blocks {
            let kind = match bd.kind.as_str() {
                "air" => BlockKind::Air,
                "solid" => BlockKind::Solid,
                "liquid" => BlockKind::Liquid,
                "foliage" => BlockKind::Foliage,
                "transparent" => BlockKind::Transparent,
                _ => BlockKind::Solid,
            };
            let textures = match &bd.textures {
                voxel_assets::BlockTexturesData::Same { same } => BlockTextures::same(*same),
                voxel_assets::BlockTexturesData::PerFace {
                    top,
                    bottom,
                    side,
                    neg_x,
                    pos_x,
                    neg_y,
                    pos_y,
                    neg_z,
                    pos_z,
                } => {
                    let t = top.unwrap_or(0);
                    let b = bottom.unwrap_or(0);
                    let s = side.unwrap_or(0);
                    // If specific faces are given, use them; otherwise fall back to top/bottom/side.
                    if neg_x.is_some()
                        || pos_x.is_some()
                        || neg_y.is_some()
                        || pos_y.is_some()
                        || neg_z.is_some()
                        || pos_z.is_some()
                    {
                        BlockTextures {
                            tiles: [
                                neg_x.unwrap_or(s),
                                pos_x.unwrap_or(s),
                                neg_y.unwrap_or(b),
                                pos_y.unwrap_or(t),
                                neg_z.unwrap_or(s),
                                pos_z.unwrap_or(s),
                            ],
                        }
                    } else {
                        BlockTextures::top_bottom_side(t, b, s)
                    }
                }
            };
            // Arc<str> owns the name; the previous global `NAME_ARENA: Mutex`
            // + `unsafe { &*ptr }` pattern was process-global mutable state
            // and broken for parallel registry construction. Owned
            // `Arc<str>` is safe, cheap to clone, and dropped with the
            // registry.
            let name: Arc<str> = Arc::from(bd.name.as_str());
            reg.add_named_owned(
                Arc::clone(&name),
                BlockDef {
                    id: BlockId(0),
                    name,
                    kind,
                    solid: bd.solid,
                    opaque: bd.opaque,
                    breakable: bd.breakable,
                    replaceable: bd.replaceable,
                    textures,
                    emission: bd.emission.min(15),
                    light_absorption: bd.light_absorption.min(15),
                },
            );
        }
        log::info!("built registry with {} blocks from assets", reg.defs.len());
        reg
    }

    fn add(&mut self, def: BlockDef) {
        let mut def = def;
        def.id = BlockId(self.defs.len() as u16);
        self.defs.push(def);
    }

    /// Insert a builtin block whose name is a string literal. The literal is
    /// interned via `Arc::from` so the resulting `BlockDef.name` shares its
    /// storage cheaply.
    fn add_named(&mut self, name: &str, mut def: BlockDef) {
        def.id = BlockId(self.defs.len() as u16);
        let arc = Arc::from(name);
        def.name = Arc::clone(&arc);
        self.by_name.insert(arc.to_string(), def.id);
        self.defs.push(def);
    }

    /// Insert a block whose name string is already owned (used by
    /// `from_assets`).
    fn add_named_owned(&mut self, name: Arc<str>, mut def: BlockDef) {
        def.id = BlockId(self.defs.len() as u16);
        def.name = Arc::clone(&name);
        self.by_name.insert(name.to_string(), def.id);
        self.defs.push(def);
    }

    pub fn get(&self, id: BlockId) -> &BlockDef {
        &self.defs[id.0 as usize]
    }

    pub fn id_of(&self, name: &str) -> Option<BlockId> {
        self.by_name.get(name).copied()
    }

    pub fn count(&self) -> usize {
        self.defs.len()
    }

    /// True if the block should be considered for collision.
    pub fn is_solid(&self, id: BlockId) -> bool {
        self.get(id).solid
    }

    /// True if the block fully hides the face of an adjacent solid block.
    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.get(id).opaque
    }

    /// True if the block is a liquid (water, lava, etc.).
    pub fn is_liquid(&self, id: BlockId) -> bool {
        self.get(id).kind == BlockKind::Liquid
    }

    /// Light emission (0–15) for the given block.
    pub fn emission(&self, id: BlockId) -> u8 {
        self.get(id).emission
    }

    /// Light absorption (0–15) for the given block.
    pub fn light_absorption(&self, id: BlockId) -> u8 {
        self.get(id).light_absorption
    }
}

fn solid_opaque(tile: u16) -> BlockDef {
    BlockDef {
        id: BlockId(0),
        name: Arc::from(""),
        kind: BlockKind::Solid,
        solid: true,
        opaque: true,
        breakable: true,
        replaceable: false,
        textures: BlockTextures::same(tile),
        emission: 0,
        light_absorption: 15,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_count() {
        let reg = BlockRegistry::with_builtins();
        assert!(
            reg.count() >= 10,
            "expected at least 10 builtins, got {}",
            reg.count()
        );
    }

    #[test]
    fn air_is_air() {
        let _reg = BlockRegistry::with_builtins();
        assert!(BlockId::AIR.is_air());
    }

    #[test]
    fn air_not_solid() {
        let reg = BlockRegistry::with_builtins();
        assert!(!reg.is_solid(BlockId::AIR));
    }

    #[test]
    fn air_not_opaque() {
        let reg = BlockRegistry::with_builtins();
        assert!(!reg.is_opaque(BlockId::AIR));
    }

    #[test]
    fn air_zero_absorption() {
        let reg = BlockRegistry::with_builtins();
        assert_eq!(reg.light_absorption(BlockId::AIR), 0);
    }

    #[test]
    fn stone_is_solid_opaque() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        assert!(!stone.is_air());
        assert!(reg.is_solid(stone));
        assert!(reg.is_opaque(stone));
        assert_eq!(reg.light_absorption(stone), 15);
    }

    #[test]
    fn torch_emits_light() {
        let reg = BlockRegistry::with_builtins();
        let torch = reg.id_of("torch").unwrap();
        assert!(reg.emission(torch) > 0);
    }

    #[test]
    fn torch_zero_absorption() {
        let reg = BlockRegistry::with_builtins();
        let torch = reg.id_of("torch").unwrap();
        assert_eq!(reg.light_absorption(torch), 0);
    }

    #[test]
    fn grass_solid_opaque() {
        let reg = BlockRegistry::with_builtins();
        let grass = reg.id_of("grass").unwrap();
        assert!(reg.is_solid(grass));
        assert!(reg.is_opaque(grass));
    }

    #[test]
    fn water_not_solid() {
        let reg = BlockRegistry::with_builtins();
        let water = reg.id_of("water").unwrap();
        assert!(!reg.is_solid(water));
    }

    #[test]
    fn id_of_unknown_returns_none() {
        let reg = BlockRegistry::with_builtins();
        assert!(reg.id_of("nonexistent_block_xyz").is_none());
    }

    #[test]
    fn block_def_texts() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        let def = reg.get(stone);
        assert_eq!(def.textures.tile(Face::PosY), def.textures.tile(Face::NegY));
    }

    #[test]
    fn is_rendered_air() {
        let reg = BlockRegistry::with_builtins();
        assert!(!reg.get(BlockId::AIR).is_rendered());
    }

    #[test]
    fn is_rendered_stone() {
        let reg = BlockRegistry::with_builtins();
        let stone = reg.id_of("stone").unwrap();
        assert!(reg.get(stone).is_rendered());
    }

    #[test]
    fn face_normal_directions() {
        assert_eq!(Face::PosX.normal(), IVec3::X);
        assert_eq!(Face::NegX.normal(), IVec3::NEG_X);
        assert_eq!(Face::PosY.normal(), IVec3::Y);
        assert_eq!(Face::NegY.normal(), IVec3::NEG_Y);
        assert_eq!(Face::PosZ.normal(), IVec3::Z);
        assert_eq!(Face::NegZ.normal(), IVec3::NEG_Z);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn id_of_get_roundtrip(id in 1u16..21) {
            let reg = BlockRegistry::with_builtins();
            let block_id = BlockId(id);
            let def = reg.get(block_id);
            let looked_up = reg.id_of(def.name.as_ref());
            prop_assert_eq!(looked_up, Some(block_id));
        }

        #[test]
        fn all_ids_in_range(id in 0u16..21) {
            let reg = BlockRegistry::with_builtins();
            let def = reg.get(BlockId(id));
            // Every block should have a non-empty name
            prop_assert!(!def.name.is_empty(), "block {id} has empty name");
        }

        #[test]
        fn emission_in_range(id in 0u16..21) {
            let reg = BlockRegistry::with_builtins();
            let e = reg.emission(BlockId(id));
            prop_assert!(e <= 15, "emission {e} out of range for block {id}");
        }

        #[test]
        fn absorption_in_range(id in 0u16..21) {
            let reg = BlockRegistry::with_builtins();
            let a = reg.light_absorption(BlockId(id));
            prop_assert!(a <= 15, "absorption {a} out of range for block {id}");
        }

        #[test]
        fn id_of_unknown_returns_none(name in "[a-z]{20,30}") {
            let reg = BlockRegistry::with_builtins();
            prop_assert_eq!(reg.id_of(&name), None);
        }
    }
}
