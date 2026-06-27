//! Per-frame input intent for the player.
//!
//! Written by `input_system` (translating the engine's resolved input into
//! a compact, system-friendly form) and read by `movement_system`.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PlayerInput {
    /// Forward / back / left / right wish direction in the XZ plane,
    /// normalised. Movement system transforms this into world space using
    /// the player's current look orientation.
    pub wish: Vec3,
    /// True if jump was pressed this frame (edge-triggered).
    pub jump: bool,
    /// True if sneak is held.
    pub sneaking: bool,
    /// True if sprint is held.
    pub sprinting: bool,
    /// True if fly toggle is active.
    pub flying: bool,
    /// Mouse delta from look, in raw units (consumed by movement system).
    pub mouse_delta: (f32, f32),
}
