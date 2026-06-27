//! Camera ownership marker + the resource that points at the local player.

use voxel_core::Camera;
use voxel_ecs::Entity;

use crate::components::player_state::PlayerState;
use crate::components::transform::Transform;

/// Marker component: this entity owns the main camera. Only one entity
/// should have this at a time.
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraOwner;

/// Resource holding the entity ID of the local player. `None` until the
/// player has been spawned.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerEntity(pub Option<Entity>);

/// Compute the camera's position and orientation from the player's
/// transform + current eye offset. Use this after `movement_system` has
/// run for the frame.
pub fn update_camera_from_transform(camera: &mut Camera, transform: &Transform, state: &PlayerState) {
    camera.pos = transform.pos + glam::Vec3::new(0.0, state.eye_offset, 0.0);
    let (yaw, pitch, _roll) = transform.rot.to_euler(glam::EulerRot::YXZ);
    camera.yaw = yaw;
    camera.pitch = pitch;
}
