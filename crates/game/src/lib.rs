//! `voxel-game` — gameplay logic: components, systems, player actions.
//!
//! The engine crate wires this into the main loop. The legacy per-struct
//! modules (`block`, `chat`, `input`, `inv`, `player`, `undo`) coexist
//! with the new ECS-based `components` and `systems`; the engine
//! integration agent will retire the legacy ones once everything is
//! ported.

pub mod block;
pub mod chat;
pub mod components;
pub mod input;
pub mod inv;
pub mod player;
pub mod systems;
pub mod undo;

pub use block::BlockAction;
pub use chat::{ChatState, CommandResult};
pub use components::{
    update_camera_from_transform, Aabb, CameraOwner, PlayerEntity, PlayerInput, PlayerState,
    Transform, Velocity,
};
pub use input::InputState;
pub use inv::Hotbar;
pub use player::{Player, PlayerConfig};
pub use systems::{
    input_system, lifecycle_system, movement_system, CameraResource, InputResource, InputSnapshot,
    PhysicsWorldRes, EYE_HEIGHT, EYE_HEIGHT_SNEAK, FLY_SPEED, GRAVITY, JUMP_SPEED,
    MOUSE_SENSITIVITY, SNEAK_SPEED, SPRINT_SPEED, SWIM_BASE_FRACTION, SWIM_UP_SPEED,
    TERMINAL_VELOCITY, WALK_SPEED, WATER_DRAG,
};
pub use undo::{BlockEdit, EditAction, UndoRedoState};
