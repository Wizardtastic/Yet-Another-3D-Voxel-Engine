//! Name-based texture atlas with PNG file loading.
//!
//! Loads a config file (`textures.toml`) from a textures directory that maps
//! tile indices to PNG filenames. Each PNG is named after what it represents
//! (stone.png, grass_top.png, etc.), not by number. This makes the system
//! scalable to hundreds of textures.
//!
//! Any tile index that is referenced by blocks but missing from the config
//! (or whose PNG file is missing) is filled with a blue+black checkerboard
//! error texture so the gap is immediately visible.

use std::collections::HashMap;
use std::path::Path;

use image::imageops::FilterType;
use voxel_core::ATLAS_TILE_SIZE;

/// Atlas side length in tiles (16×16 = 256 tiles). Single source of truth.
pub use voxel_core::ATLAS_TILES;
/// Atlas side length in pixels.
pub const ATLAS_PIXELS: u32 = ATLAS_TILES * ATLAS_TILE_SIZE;

/// A finished atlas: RGBA8 pixels ready to upload to a Vulkan image.
pub struct Atlas {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Write a pixel into a tile at tile-local (tx, ty).
#[allow(clippy::too_many_arguments)]
fn put(atlas: &mut [u8], tile: u32, tx: u32, ty: u32, r: u8, g: u8, b: u8, a: u8) {
    let tile_x = (tile % ATLAS_TILES) * ATLAS_TILE_SIZE;
    let tile_y = (tile / ATLAS_TILES) * ATLAS_TILE_SIZE;
    let px = tile_x + tx;
    let py = tile_y + ty;
    let idx = ((py * ATLAS_PIXELS + px) * 4) as usize;
    atlas[idx] = r;
    atlas[idx + 1] = g;
    atlas[idx + 2] = b;
    atlas[idx + 3] = a;
}

/// Write every pixel of a tile using the supplied closure.
fn fill_tile<F: Fn(u32, u32) -> (u8, u8, u8, u8)>(atlas: &mut [u8], tile: u32, f: F) {
    for ty in 0..ATLAS_TILE_SIZE {
        for tx in 0..ATLAS_TILE_SIZE {
            let (r, g, b, a) = f(tx, ty);
            put(atlas, tile, tx, ty, r, g, b, a);
        }
    }
}

/// Fill a tile with the blue+black checkerboard error pattern.
fn fill_error_tile(atlas: &mut [u8], tile: u32) {
    fill_tile(atlas, tile, |x, y| {
        // 2×2 pixel checkerboard: alternating blue and black.
        let checker = ((x / 2) + (y / 2)) % 2 == 0;
        if checker {
            (0, 60, 220, 255) // blue
        } else {
            (0, 0, 0, 255) // black
        }
    });
}

/// Build the atlas: fill every tile with the error texture, then overlay
/// PNGs loaded from `textures_dir/textures.toml` config. Tiles not present
/// in the config keep the error texture.
pub fn build_atlas_with_textures(textures_dir: &Path) -> Atlas {
    let total_tiles = (ATLAS_TILES * ATLAS_TILES) as usize;
    let mut rgba = vec![0u8; (ATLAS_PIXELS * ATLAS_PIXELS * 4) as usize];

    // Fill every tile with the error texture first.
    for tile in 0..total_tiles as u32 {
        fill_error_tile(&mut rgba, tile);
    }

    let mut atlas = Atlas {
        width: ATLAS_PIXELS,
        height: ATLAS_PIXELS,
        rgba,
    };

    let mapping = load_texture_config(textures_dir);
    let mut loaded = 0;
    let mut missing = Vec::new();
    for (tile_index, filename) in &mapping {
        if *tile_index >= total_tiles as u32 {
            log::warn!(
                "texture config: tile index {} out of range (0..{}), skipping '{}'",
                tile_index,
                total_tiles,
                filename
            );
            continue;
        }
        let png_path = textures_dir.join(filename);
        match load_png_into_tile(&mut atlas, *tile_index, &png_path) {
            Ok(()) => {
                loaded += 1;
            }
            Err(e) => {
                log::warn!("failed to load texture {}: {}", png_path.display(), e);
                missing.push(*tile_index);
            }
        }
    }

    if loaded > 0 {
        log::info!(
            "loaded {}/{} textures from {}",
            loaded,
            mapping.len(),
            textures_dir.display()
        );
    }
    if !missing.is_empty() {
        log::warn!(
            "{} texture(s) still missing - will show error pattern: {:?}",
            missing.len(),
            missing
        );
    }
    if mapping.is_empty() {
        log::warn!(
            "no textures.toml found in {} — all tiles show error pattern",
            textures_dir.display()
        );
    }

    atlas
}

/// Read `textures.toml` from `textures_dir` and return a map of
/// tile_index → png_filename. The config format is:
///
/// ```toml
/// [tiles]
/// 0 = "air.png"
/// 1 = "stone.png"
/// 2 = "dirt.png"
/// ```
///
/// Returns an empty map if the file doesn't exist or can't be parsed.
fn load_texture_config(textures_dir: &Path) -> HashMap<u32, String> {
    let config_path = textures_dir.join("textures.toml");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(s) => s,
        Err(_) => return HashMap::new(),
    };
    let value: toml::Value = match content.parse() {
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "failed to parse {}: {}",
                config_path.display(),
                e
            );
            return HashMap::new();
        }
    };
    let tiles = match value.get("tiles").and_then(|v| v.as_table()) {
        Some(t) => t,
        None => {
            log::warn!(
                "no [tiles] section in {}",
                config_path.display()
            );
            return HashMap::new();
        }
    };
    let mut map = HashMap::new();
    for (key, val) in tiles {
        let Ok(tile_index) = key.parse::<u32>() else {
            log::warn!("texture config: invalid tile index '{}'", key);
            continue;
        };
        let Some(filename) = val.as_str() else {
            log::warn!(
                "texture config: tile {} value must be a string filename",
                tile_index
            );
            continue;
        };
        map.insert(tile_index, filename.to_string());
    }
    map
}

/// Decode a PNG file and write its pixels into the given tile in the atlas.
/// The image is resized to ATLAS_TILE_SIZE x ATLAS_TILE_SIZE.
fn load_png_into_tile(
    atlas: &mut Atlas,
    tile: u32,
    path: &Path,
) -> Result<(), String> {
    let img = image::open(path).map_err(|e| format!("decode: {e}"))?;
    let resized = img.resize_exact(
        ATLAS_TILE_SIZE,
        ATLAS_TILE_SIZE,
        FilterType::Nearest,
    );
    let rgba_img = resized.to_rgba8();
    let tile_x = (tile % ATLAS_TILES) * ATLAS_TILE_SIZE;
    let tile_y = (tile / ATLAS_TILES) * ATLAS_TILE_SIZE;
    for ty in 0..ATLAS_TILE_SIZE {
        for tx in 0..ATLAS_TILE_SIZE {
            let pixel = rgba_img.get_pixel(tx, ty);
            let px = tile_x + tx;
            let py = tile_y + ty;
            let idx = ((py * atlas.width + px) * 4) as usize;
            atlas.rgba[idx] = pixel[0];
            atlas.rgba[idx + 1] = pixel[1];
            atlas.rgba[idx + 2] = pixel[2];
            atlas.rgba[idx + 3] = pixel[3];
        }
    }
    Ok(())
}
