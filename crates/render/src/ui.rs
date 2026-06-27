//! UI overlay system: bitmap font, vertex format, and draw-data builder.
//!
//! The engine builds a `UiDrawData` each frame (crosshair, hotbar, pause menu)
//! and passes it to `Renderer::draw_frame`. The renderer uploads the vertices
//! to a persistent host-visible buffer and draws them with the UI pipeline
//! after the chunk pass.

use bytemuck::{Pod, Zeroable};
use voxel_core::ATLAS_TILE_SIZE;

use crate::atlas::{Atlas, ATLAS_TILES};

// ── Vertex format ───────────────────────────────────────────────────────

/// UI vertex: 24 bytes. Screen-space pixel coords, atlas UVs, colour tint,
/// and a texture selector (0 = block atlas, 1 = font atlas).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [u8; 4],
    pub tex_id: f32,
}

/// Collected UI vertices + indices for one frame.
#[derive(Clone, Default, Debug)]
pub struct UiDrawData {
    pub vertices: Vec<UiVertex>,
    pub indices: Vec<u32>,
}

impl UiDrawData {
    /// Push a coloured quad (uses the white tile in the block atlas, tex_id=0).
    pub fn quad(&mut self, x: f32, y: f32, w: f32, h: f32, color: [u8; 4]) {
        let tile = 20u32; // white tile
        let tx = tile % ATLAS_TILES;
        let ty = tile / ATLAS_TILES;
        let u0 = tx as f32 / ATLAS_TILES as f32;
        let v0 = ty as f32 / ATLAS_TILES as f32;
        let u1 = (tx + 1) as f32 / ATLAS_TILES as f32;
        let v1 = (ty + 1) as f32 / ATLAS_TILES as f32;
        self.quad_uv(x, y, w, h, u0, v0, u1, v1, color, 0.0);
    }

    /// Push a quad sampling a specific tile from the block atlas (tex_id=0).
    pub fn block_icon(&mut self, x: f32, y: f32, w: f32, h: f32, tile: u16, color: [u8; 4]) {
        let tx = tile as u32 % ATLAS_TILES;
        let ty = tile as u32 / ATLAS_TILES;
        let u0 = tx as f32 / ATLAS_TILES as f32;
        let v0 = ty as f32 / ATLAS_TILES as f32;
        let u1 = (tx + 1) as f32 / ATLAS_TILES as f32;
        let v1 = (ty + 1) as f32 / ATLAS_TILES as f32;
        self.quad_uv(x, y, w, h, u0, v0, u1, v1, color, 0.0);
    }

    /// Push a quad with explicit UVs and tex_id.
    #[allow(clippy::too_many_arguments)]
    fn quad_uv(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        u0: f32,
        v0: f32,
        u1: f32,
        v1: f32,
        color: [u8; 4],
        tex_id: f32,
    ) {
        let start = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&[
            UiVertex {
                pos: [x, y],
                uv: [u0, v0],
                color,
                tex_id,
            },
            UiVertex {
                pos: [x + w, y],
                uv: [u1, v0],
                color,
                tex_id,
            },
            UiVertex {
                pos: [x + w, y + h],
                uv: [u1, v1],
                color,
                tex_id,
            },
            UiVertex {
                pos: [x, y + h],
                uv: [u0, v1],
                color,
                tex_id,
            },
        ]);
        self.indices
            .extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    }

    /// Push a text string using the bitmap font (tex_id=1).
    pub fn text(
        &mut self,
        s: &str,
        x: f32,
        y: f32,
        scale: f32,
        color: [u8; 4],
        font: &FontAtlas,
    ) -> f32 {
        let mut cx = x;
        for ch in s.chars() {
            if ch == ' ' {
                cx += FONT_ADVANCE * scale;
                continue;
            }
            if let Some((u0, v0, u1, v1)) = font.char_uv(ch) {
                let w = FONT_ADVANCE * scale;
                let h = FONT_HEIGHT * scale;
                self.quad_uv(cx, y, w, h, u0, v0, u1, v1, color, 1.0);
                cx += w;
            }
        }
        cx
    }

    /// Push a hollow rectangle (4 thin quads forming a border).
    pub fn rect_border(&mut self, x: f32, y: f32, w: f32, h: f32, thickness: f32, color: [u8; 4]) {
        self.quad(x, y, w, thickness, color); // top
        self.quad(x, y + h - thickness, w, thickness, color); // bottom
        self.quad(x, y, thickness, h, color); // left
        self.quad(x + w - thickness, y, thickness, h, color); // right
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

// ── Bitmap font ─────────────────────────────────────────────────────────

/// Pixel advance per character (the 5-wide bitmap + 1px spacing, scaled to
/// fill most of a 16px tile → 12px advance).
const FONT_ADVANCE: f32 = 12.0;
/// Pixel height of a character quad.
const FONT_HEIGHT: f32 = 16.0;
/// Font atlas tile grid (columns × rows).
const FONT_COLS: u32 = 16;
const FONT_ROWS: u32 = 3;
/// Font atlas pixel size.
const FONT_ATLAS_W: u32 = FONT_COLS * ATLAS_TILE_SIZE;
const FONT_ATLAS_H: u32 = FONT_ROWS * ATLAS_TILE_SIZE;

/// 5×7 bitmap font for A–Z, 0–9, punctuation, and space. Each entry is 7 bytes;
/// low 5 bits of each byte = one row (MSB = leftmost pixel).
const FONT_DATA: &[(char, [u8; 7])] = &[
    ('A', [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11]),
    ('B', [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E]),
    ('C', [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E]),
    ('D', [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C]),
    ('E', [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F]),
    ('F', [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10]),
    ('G', [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E]),
    ('H', [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11]),
    ('I', [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F]),
    ('J', [0x01, 0x01, 0x01, 0x01, 0x01, 0x11, 0x0E]),
    ('K', [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11]),
    ('L', [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F]),
    ('M', [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11]),
    ('N', [0x11, 0x19, 0x15, 0x15, 0x13, 0x11, 0x11]),
    ('O', [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E]),
    ('P', [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10]),
    ('Q', [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D]),
    ('R', [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11]),
    ('S', [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E]),
    ('T', [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04]),
    ('U', [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E]),
    ('V', [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04]),
    ('W', [0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11]),
    ('X', [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11]),
    ('Y', [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04]),
    ('Z', [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F]),
    ('0', [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E]),
    ('1', [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E]),
    ('2', [0x0E, 0x11, 0x01, 0x06, 0x08, 0x10, 0x1F]),
    ('3', [0x0E, 0x11, 0x01, 0x06, 0x01, 0x11, 0x0E]),
    ('4', [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02]),
    ('5', [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E]),
    ('6', [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E]),
    ('7', [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08]),
    ('8', [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E]),
    ('9', [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C]),
    ('.', [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C]),
    (':', [0x00, 0x0C, 0x0C, 0x00, 0x0C, 0x0C, 0x00]),
    ('/', [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10]),
    ('-', [0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00]),
    ('+', [0x00, 0x04, 0x04, 0x1F, 0x04, 0x04, 0x00]),
    ('[', [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E]),
    (']', [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E]),
];

/// Runtime font atlas: pixel data + character→UV lookup.
pub struct FontAtlas {
    pub atlas: Atlas,
    /// Maps character → tile index in the font atlas.
    char_map: std::collections::HashMap<char, u32>,
}

impl Default for FontAtlas {
    fn default() -> Self {
        Self::new()
    }
}

impl FontAtlas {
    pub fn new() -> Self {
        let mut rgba = vec![0u8; (FONT_ATLAS_W * FONT_ATLAS_H * 4) as usize];
        let mut char_map = std::collections::HashMap::new();

        for (i, &(ch, bitmap)) in FONT_DATA.iter().enumerate() {
            let tile = i as u32;
            char_map.insert(ch, tile);
            let tile_x = (tile % FONT_COLS) * ATLAS_TILE_SIZE;
            let tile_y = (tile / FONT_COLS) * ATLAS_TILE_SIZE;
            // Render the 5×7 bitmap scaled 2× (→10×14) and centred in 16×16.
            let off_x = (ATLAS_TILE_SIZE as i32 - 10) / 2; // 3
            let off_y = (ATLAS_TILE_SIZE as i32 - 14) / 2; // 1
            for row in 0..7i32 {
                for col in 0..5i32 {
                    let bit = (bitmap[row as usize] >> (4 - col)) & 1;
                    if bit != 0 {
                        for dy in 0..2i32 {
                            for dx in 0..2i32 {
                                let px = tile_x as i32 + off_x + col * 2 + dx;
                                let py = tile_y as i32 + off_y + row * 2 + dy;
                                if (tile_x as i32..(tile_x + ATLAS_TILE_SIZE) as i32).contains(&px)
                                    && (tile_y as i32..(tile_y + ATLAS_TILE_SIZE) as i32).contains(&py)
                                {
                                    let idx = ((py as u32 * FONT_ATLAS_W + px as u32) * 4) as usize;
                                    rgba[idx] = 255;
                                    rgba[idx + 1] = 255;
                                    rgba[idx + 2] = 255;
                                    rgba[idx + 3] = 255;
                                }
                            }
                        }
                    }
                }
            }
        }

        let atlas = Atlas {
            width: FONT_ATLAS_W,
            height: FONT_ATLAS_H,
            rgba,
        };
        Self { atlas, char_map }
    }

    /// UV coordinates for a character in the font atlas, or None if unsupported.
    pub fn char_uv(&self, ch: char) -> Option<(f32, f32, f32, f32)> {
        let upper = ch.to_ascii_uppercase();
        let tile = *self.char_map.get(&upper)?;
        let tx = tile % FONT_COLS;
        let ty = tile / FONT_COLS;
        let u0 = tx as f32 / FONT_COLS as f32;
        let v0 = ty as f32 / FONT_ROWS as f32;
        let u1 = (tx + 1) as f32 / FONT_COLS as f32;
        let v1 = (ty + 1) as f32 / FONT_ROWS as f32;
        Some((u0, v0, u1, v1))
    }

    /// Total width of a text string at the given scale.
    /// Only counts characters that are spaces or exist in the font atlas,
    /// matching the behavior of `Ui::text()`.
    pub fn text_width(&self, s: &str, scale: f32) -> f32 {
        let mut count = 0.0f32;
        for ch in s.chars() {
            if ch == ' ' || self.char_uv(ch).is_some() {
                count += 1.0;
            }
        }
        count * FONT_ADVANCE * scale
    }
}
