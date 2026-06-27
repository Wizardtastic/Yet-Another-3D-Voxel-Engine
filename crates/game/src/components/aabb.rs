//! Half-extent axis-aligned bounding box. The full AABB is
//! `[-half, +half]` around the entity's transform origin.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Aabb {
    pub half: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Self {
            half: Vec3::new(0.3, 0.9, 0.3),
        }
    }
}
