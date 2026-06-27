//! Linear + angular velocity. `lin` is in m/s, `ang` is in rad/s around the
//! entity's local axes.

use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Velocity {
    pub lin: Vec3,
    pub ang: Vec3,
}
