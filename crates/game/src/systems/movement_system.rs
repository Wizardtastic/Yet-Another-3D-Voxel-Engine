//! Applies movement physics to the player entity (Transform + Velocity +
//! Aabb + PlayerInput + PlayerState). Supports fly / water / normal modes
//! with full swept-AABB collision against the voxel world.

use voxel_core::Camera;
use voxel_ecs::World;

use crate::components::{PlayerEntity, PlayerInput, PlayerState, Transform, Velocity, Aabb};

/// Resource: shared camera. The engine writes/reads it; the movement
/// system doesn't touch it directly (use `update_camera_from_transform`).
#[derive(Clone, Copy, Debug, Default)]
pub struct CameraResource(pub Camera);

/// Resource: the voxel world, shared from the engine. The engine inserts
/// this every frame before running gameplay systems. The movement system
/// uses it for collision detection and water checks.
#[derive(Clone)]
pub struct PhysicsWorldRes(pub std::sync::Arc<voxel_world::world::World>);

pub const WALK_SPEED: f32 = 4.317;
pub const SPRINT_SPEED: f32 = 5.6;
pub const SNEAK_SPEED: f32 = 1.8;
pub const JUMP_SPEED: f32 = 7.5;
pub const GRAVITY: f32 = 26.0;
pub const TERMINAL_VELOCITY: f32 = 55.0;
pub const FLY_SPEED: f32 = 20.0;
pub const SWIM_UP_SPEED: f32 = 4.0;
pub const EYE_HEIGHT: f32 = 0.8;
pub const EYE_HEIGHT_SNEAK: f32 = 0.55;
pub const MOUSE_SENSITIVITY: f32 = 0.0022;
pub const WATER_DRAG: f32 = 0.7;
pub const SWIM_BASE_FRACTION: f32 = 0.5;

const SNEAK_AABB_Y: f32 = 0.65;
const PLAYER_AABB_XZ: f32 = 0.3;
const PLAYER_AABB_Y: f32 = 0.9;

/// System: applies movement physics to the player entity for one frame.
pub fn movement_system(world: &mut World, dt: f32) {
    let player_entity = match world.resource::<PlayerEntity>().and_then(|p| p.0) {
        Some(e) => e,
        None => return,
    };

    // Snapshot the components we need.
    let initial = {
        let transform = match world.get::<Transform>(player_entity) {
            Some(t) => *t,
            None => return,
        };
        let velocity = match world.get::<Velocity>(player_entity) {
            Some(v) => *v,
            None => return,
        };
        let aabb = match world.get::<Aabb>(player_entity) {
            Some(a) => *a,
            None => Aabb::default(),
        };
        let input = match world.get::<PlayerInput>(player_entity) {
            Some(i) => *i,
            None => return,
        };
        let state = match world.get::<PlayerState>(player_entity) {
            Some(s) => *s,
            None => return,
        };
        (transform, velocity, aabb, input, state)
    };

    let (mut transform, velocity, aabb_base, input, mut state) = initial;

    // Mouse look: subtract delta (matches the original Player::update).
    if input.mouse_delta.0 != 0.0 || input.mouse_delta.1 != 0.0 {
        let (mut yaw, mut pitch, _roll) = transform.rot.to_euler(glam::EulerRot::YXZ);
        yaw -= input.mouse_delta.0 * MOUSE_SENSITIVITY;
        pitch -= input.mouse_delta.1 * MOUSE_SENSITIVITY;
        let max_pitch = std::f32::consts::FRAC_PI_2 - 0.01;
        pitch = pitch.clamp(-max_pitch, max_pitch);
        transform.rot = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
    }

    // Build a world-space wish direction from the player's yaw/pitch.
    // Convention: forward = (-sin(yaw)*cos(pitch), -sin(pitch), -cos(yaw)*cos(pitch)),
    // right = (cos(yaw), 0, -sin(yaw)). This matches the Camera struct.
    let (yaw, pitch, _roll) = transform.rot.to_euler(glam::EulerRot::YXZ);
    let cp = pitch.cos();
    let forward = glam::Vec3::new(-yaw.sin() * cp, -pitch.sin(), -yaw.cos() * cp);
    let right_flat = glam::Vec3::new(yaw.cos(), 0.0, -yaw.sin());

    // Build a horizontal wish direction from the input components.
    // InputSystem already stored the wish normalised with z=-1 for forward.
    let mut wish = glam::Vec3::ZERO;
    if input.wish.z < 0.0 {
        wish += forward * -input.wish.z;
    }
    if input.wish.z > 0.0 {
        wish -= forward * input.wish.z;
    }
    if input.wish.x < 0.0 {
        wish -= right_flat * -input.wish.x;
    }
    if input.wish.x > 0.0 {
        wish += right_flat * input.wish.x;
    }
    // Flatten the wish to the XZ plane so looking up/down doesn't reduce
    // horizontal movement speed.
    wish.y = 0.0;
    // Normalise the horizontal wish.
    let wish_horizontal = if wish.length_squared() > 0.0 {
        wish.normalize()
    } else {
        glam::Vec3::ZERO
    };

    let speed = if input.sneaking {
        SNEAK_SPEED
    } else if input.sprinting {
        SPRINT_SPEED
    } else {
        WALK_SPEED
    };

    // Check for water in the player's AABB.
    let physics = world.resource::<PhysicsWorldRes>().cloned();
    let in_water = physics
        .as_ref()
        .map(|w| {
            let half = glam::Vec3::new(
                PLAYER_AABB_XZ,
                if input.sneaking { SNEAK_AABB_Y } else { PLAYER_AABB_Y },
                PLAYER_AABB_XZ,
            );
            let aabb_min = transform.pos - half;
            let aabb_max = transform.pos + half;
            let min_b = voxel_core::math::world_to_block(aabb_min);
            let max_b = voxel_core::math::world_to_block(aabb_max - glam::Vec3::splat(0.001));
            for by in min_b.y..=max_b.y {
                for bz in min_b.z..=max_b.z {
                    for bx in min_b.x..=max_b.x {
                        if w.0.is_liquid(bx, by, bz) {
                            return true;
                        }
                    }
                }
            }
            false
        })
        .unwrap_or(false);

    // Compute new velocity based on mode.
    let mut new_vel = velocity.lin;

    if input.flying {
        // Fly mode: direct velocity from input.
        let mut fly_vel = wish_horizontal * FLY_SPEED;
        fly_vel.y = 0.0;
        if input.jump {
            fly_vel.y += FLY_SPEED;
        }
        if input.sneaking {
            fly_vel.y -= FLY_SPEED;
        }
        new_vel = fly_vel;
    } else if in_water {
        new_vel *= 1.0 - WATER_DRAG * dt;
        if input.jump {
            new_vel.y = SWIM_UP_SPEED;
        } else if input.sneaking {
            new_vel.y = -SWIM_UP_SPEED;
        } else {
            new_vel.y -= GRAVITY * 0.15 * dt;
        }
        new_vel.y = new_vel.y.max(-3.0);
        if wish_horizontal.length_squared() > 0.0 {
            let target = wish_horizontal * (speed * SWIM_BASE_FRACTION);
            new_vel.x = approach(new_vel.x, target.x, dt * 10.0);
            new_vel.z = approach(new_vel.z, target.z, dt * 10.0);
        }
    } else {
        // Normal mode: gravity, jump, ground movement.
        new_vel.y -= GRAVITY * dt;
        new_vel.y = new_vel.y.max(-TERMINAL_VELOCITY);
        if state.on_ground && input.jump {
            new_vel.y = JUMP_SPEED;
            state.on_ground = false;
        }
        if wish_horizontal.length_squared() > 1e-6 {
            let target = wish_horizontal * speed;
            new_vel.x = approach(new_vel.x, target.x, dt * 20.0);
            new_vel.z = approach(new_vel.z, target.z, dt * 20.0);
        } else if state.on_ground {
            new_vel.x *= 0.0; // stop on ground
            new_vel.z *= 0.0;
        }
    }

    // Track fall speed for landing-damage style effects in other systems.
    if new_vel.y < state.fall_speed_peak {
        state.fall_speed_peak = new_vel.y;
    }

    // --- collision: swept AABB against the voxel world ---
    let aabb_half = glam::Vec3::new(
        aabb_base.half.x,
        if input.sneaking { SNEAK_AABB_Y } else { aabb_base.half.y },
        aabb_base.half.z,
    );
    let mut new_pos = transform.pos;
    let mut landed = false;

    if let Some(phys) = &physics {
        let res = voxel_physics::swept_aabb(
            &phys.0,
            transform.pos,
            aabb_half,
            new_vel * dt,
        );
        new_pos = res.new_pos;
        // Zero velocity on blocked axes.
        if res.hit[0] || res.hit[1] {
            new_vel.x = 0.0;
        }
        if res.hit[2] || res.hit[3] {
            new_vel.y = 0.0;
        }
        if res.hit[4] || res.hit[5] {
            new_vel.z = 0.0;
        }
        landed = !state.on_ground && res.on_ground;
        state.on_ground = res.on_ground;
    } else {
        // No physics world: move freely.
        new_pos = transform.pos + new_vel * dt;
    }

    if landed {
        state.fall_speed_peak = 0.0;
    }
    if state.on_ground && new_vel.y <= 0.0 {
        state.fall_speed_peak = 0.0;
    }

    state.eye_offset = if input.sneaking { EYE_HEIGHT_SNEAK } else { EYE_HEIGHT };
    state.was_in_water = state.in_water;
    state.in_water = in_water;

    transform.pos = new_pos;

    // Commit each component separately to avoid overlapping &mut borrows.
    world.set(player_entity, transform);
    world.set(player_entity, Velocity { lin: new_vel, ang: glam::Vec3::ZERO });
    world.set(player_entity, state);
}

/// Smoothly approach a target value with a maximum rate of change.
fn approach(current: f32, target: f32, max_delta: f32) -> f32 {
    if current < target {
        (current + max_delta).min(target)
    } else {
        (current - max_delta).max(target)
    }
}
