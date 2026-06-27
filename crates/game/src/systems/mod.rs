//! Gameplay systems: input -> movement -> lifecycle.
//!
//! Each system is a plain `fn(&mut World, f32)` so the engine can wrap it
//! in an `FnSystem` and add it to a `SystemSchedule`.

mod input_system;
mod lifecycle_system;
mod movement_system;

pub use input_system::{input_system, InputResource, InputSnapshot};
pub use lifecycle_system::lifecycle_system;
pub use movement_system::{
    movement_system, CameraResource, PhysicsWorldRes, EYE_HEIGHT, EYE_HEIGHT_SNEAK, FLY_SPEED,
    GRAVITY, JUMP_SPEED, MOUSE_SENSITIVITY, SNEAK_SPEED, SPRINT_SPEED, SWIM_BASE_FRACTION,
    SWIM_UP_SPEED, TERMINAL_VELOCITY, WALK_SPEED, WATER_DRAG,
};
