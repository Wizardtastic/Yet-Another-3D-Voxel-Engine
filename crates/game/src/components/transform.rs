//! World-space position + orientation for any entity that lives in the world.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Transform {
    pub pos: Vec3,
    pub rot: Quat,
}
