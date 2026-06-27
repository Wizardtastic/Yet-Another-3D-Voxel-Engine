//! ECS components used by gameplay systems.
//!
//! These are small data containers attached to entities. Logic lives in
//! `crate::systems`; data lives here.

mod aabb;
mod camera_owner;
mod player_input;
mod player_state;
mod transform;
mod velocity;

pub use aabb::Aabb;
pub use camera_owner::{update_camera_from_transform, CameraOwner, PlayerEntity};
pub use player_input::PlayerInput;
pub use player_state::PlayerState;
pub use transform::Transform;
pub use velocity::Velocity;
