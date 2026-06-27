//! Persistent player state across frames. Written by `movement_system`,
//! read by other systems (camera, audio, particles, ...).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct PlayerState {
    pub on_ground: bool,
    pub in_water: bool,
    pub was_in_water: bool,
    pub fall_speed_peak: f32,
    /// Y offset for the camera (0.8 standing, 0.55 sneaking).
    pub eye_offset: f32,
}
