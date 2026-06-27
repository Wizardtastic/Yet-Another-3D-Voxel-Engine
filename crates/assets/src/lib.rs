//! `voxel-assets` — data-driven content + asset loading framework.
//!
//! Content (blocks, items, biomes, recipes, entities) is defined in data files,
//! not code, which is the foundation for the modding API. Responsibilities:
//! - block/item/entity registries populated at startup from JSON/asset packs
//! - texture atlas packing from PNG sources
//! - recipe + loot tables
//! - the plugin/event surface (`EventBus`, `Registry` hooks) used by mods

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde::de;

/// Serde-friendly representation of a block type loaded from JSON.
#[derive(Debug, Deserialize)]
pub struct BlockData {
    pub name: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default = "default_true")]
    pub solid: bool,
    #[serde(default = "default_true")]
    pub opaque: bool,
    #[serde(default = "default_true")]
    pub breakable: bool,
    #[serde(default = "default_replaceable")]
    pub replaceable: bool,
    #[serde(default)]
    pub textures: BlockTexturesData,
    #[serde(default)]
    pub emission: u8,
    #[serde(default = "default_absorption")]
    pub light_absorption: u8,
}

#[derive(Debug)]
pub enum BlockTexturesData {
    /// All six faces share one tile index.
    Same { same: u16 },
    /// Per-face specification.
    PerFace {
        top: Option<u16>,
        bottom: Option<u16>,
        side: Option<u16>,
        neg_x: Option<u16>,
        pos_x: Option<u16>,
        neg_y: Option<u16>,
        pos_y: Option<u16>,
        neg_z: Option<u16>,
        pos_z: Option<u16>,
    },
}

#[derive(Debug, Default, Deserialize)]
struct PerFaceFields {
    #[serde(default)]
    top: Option<u16>,
    #[serde(default)]
    bottom: Option<u16>,
    #[serde(default)]
    side: Option<u16>,
    #[serde(default)]
    neg_x: Option<u16>,
    #[serde(default)]
    pos_x: Option<u16>,
    #[serde(default)]
    neg_y: Option<u16>,
    #[serde(default)]
    pos_y: Option<u16>,
    #[serde(default)]
    neg_z: Option<u16>,
    #[serde(default)]
    pos_z: Option<u16>,
}

impl<'de> Deserialize<'de> for BlockTexturesData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        if let Some(same) = value.get("same") {
            if same.is_u64() {
                let same_val = same.as_u64().unwrap_or(0) as u16;
                return Ok(BlockTexturesData::Same { same: same_val });
            } else if same.is_i64() {
                let v = same.as_i64().unwrap_or(-1);
                if v >= 0 && v <= u16::MAX as i64 {
                    return Ok(BlockTexturesData::Same { same: v as u16 });
                }
                return Err(de::Error::custom("field 'same' must be a number in 0..=65535"));
            } else {
                return Err(de::Error::custom("field 'same' must be a number"));
            }
        }
        let pf: PerFaceFields =
            serde_json::from_value(value).map_err(de::Error::custom)?;
        Ok(BlockTexturesData::PerFace {
            top: pf.top,
            bottom: pf.bottom,
            side: pf.side,
            neg_x: pf.neg_x,
            pos_x: pf.pos_x,
            neg_y: pf.neg_y,
            pos_y: pf.pos_y,
            neg_z: pf.neg_z,
            pos_z: pf.pos_z,
        })
    }
}

impl Default for BlockTexturesData {
    fn default() -> Self {
        BlockTexturesData::Same { same: 0 }
    }
}

fn default_kind() -> String {
    "solid".into()
}
fn default_true() -> bool {
    true
}
fn default_replaceable() -> bool {
    false
}
fn default_absorption() -> u8 {
    15
}

/// Loads block definitions from a JSON file.
pub struct AssetLoader {
    blocks_dir: PathBuf,
}

impl AssetLoader {
    pub fn new(blocks_dir: impl Into<PathBuf>) -> Self {
        Self {
            blocks_dir: blocks_dir.into(),
        }
    }

    /// Load all block definitions from `{blocks_dir}/blocks.json`.
    pub fn load_blocks(&self) -> Result<Vec<BlockData>> {
        let path = self.blocks_dir.join("blocks.json");
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let blocks: Vec<BlockData> = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        log::info!(
            "loaded {} block definitions from {}",
            blocks.len(),
            path.display()
        );
        Ok(blocks)
    }

    /// Check if the blocks directory exists and has a blocks.json file.
    pub fn has_blocks(&self) -> bool {
        self.blocks_dir.join("blocks.json").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_block_json() {
        let json = r#"
        [
            {
                "name": "stone",
                "kind": "solid",
                "solid": true,
                "opaque": true,
                "textures": { "same": 1 },
                "emission": 0,
                "light_absorption": 15
            },
            {
                "name": "grass",
                "textures": { "top": 3, "bottom": 2, "side": 4 }
            },
            {
                "name": "air",
                "kind": "air",
                "solid": false,
                "opaque": false,
                "emission": 0,
                "light_absorption": 0
            }
        ]
        "#;
        let blocks: Vec<BlockData> = serde_json::from_str(json).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0].name, "stone");
        assert!(blocks[0].solid);
        assert_eq!(blocks[0].light_absorption, 15);
        // grass uses defaults
        assert_eq!(blocks[1].name, "grass");
        assert!(blocks[1].solid); // default
        assert_eq!(blocks[1].light_absorption, 15); // default
    }

    #[test]
    fn parse_textures_same() {
        let json = r#"{ "same": 5 }"#;
        let t: BlockTexturesData = serde_json::from_str(json).unwrap();
        match t {
            BlockTexturesData::Same { same } => assert_eq!(same, 5),
            _ => panic!("expected Same"),
        }
    }

    #[test]
    fn parse_textures_per_face() {
        let json = r#"{ "top": 1, "bottom": 2, "side": 3 }"#;
        let t: BlockTexturesData = serde_json::from_str(json).unwrap();
        match t {
            BlockTexturesData::PerFace {
                top,
                bottom,
                side,
                neg_x,
                ..
            } => {
                assert_eq!(top, Some(1));
                assert_eq!(bottom, Some(2));
                assert_eq!(side, Some(3));
                assert_eq!(neg_x, None);
            }
            _ => panic!("expected PerFace"),
        }
    }
}
