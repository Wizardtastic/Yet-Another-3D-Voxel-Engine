//! First-person player controller.
//!
//! Updates look angles from mouse delta, integrates velocity with gravity, and
//! moves the player's AABB through the voxel grid using `voxel-physics`. Handles
//! walking, sprinting, sneaking (slower + reduced height), jumping, and basic
//! fall damage on hard landings (foundation; health system hooks added later).

use voxel_core::{math::world_to_block, Camera};
use voxel_physics::{intersects_solid, swept_aabb};
use voxel_world::World;

use crate::input::{Action, InputState};

/// Player dimensions (half-extents of the AABB, in blocks/metres).
pub const PLAYER_HALF: glam::Vec3 = glam::Vec3::new(0.3, 0.9, 0.3);
/// Sneak height reduces the half-extent Y.
pub const PLAYER_HALF_SNEAK_Y: f32 = 0.65;
/// Eye height above the AABB centre.
pub const EYE_HEIGHT: f32 = 0.8;

#[derive(Clone, Copy, Debug, serde::Deserialize)]
#[serde(default)]
pub struct PlayerConfig {
    pub walk_speed: f32,
    pub sprint_speed: f32,
    pub sneak_speed: f32,
    pub jump_speed: f32,
    pub gravity: f32,
    /// Max fall speed (terminal velocity).
    pub terminal_velocity: f32,
    pub mouse_sensitivity: f32,
    pub fly_speed: f32,
}

impl Default for PlayerConfig {
    fn default() -> Self {
        Self {
            walk_speed: 4.317, // MC walking m/s
            sprint_speed: 5.6,
            sneak_speed: 1.8,
            jump_speed: 7.5,
            gravity: 26.0,
            terminal_velocity: 55.0,
            mouse_sensitivity: 0.0022,
            fly_speed: 20.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Player {
    pub camera: Camera,
    /// AABB centre position in world space.
    pub pos: glam::Vec3,
    pub vel: glam::Vec3,
    pub on_ground: bool,
    pub sneaking: bool,
    pub sprinting: bool,
    pub flying: bool,
    pub config: PlayerConfig,
    /// Accumulated downward speed for fall-damage on landing.
    fall_speed_peak: f32,
    /// True when the player is submerged in water.
    pub in_water: bool,
    /// Previous frame's water state (for transition detection).
    was_in_water: bool,
}

impl Player {
    /// Spawn at `pos` (AABB centre). Looks horizontally by default.
    pub fn new(pos: glam::Vec3, config: PlayerConfig) -> Self {
        let camera = Camera {
            pos: pos + glam::Vec3::new(0.0, EYE_HEIGHT, 0.0),
            ..Default::default()
        };
        Self {
            camera,
            pos,
            vel: glam::Vec3::ZERO,
            on_ground: false,
            sneaking: false,
            sprinting: false,
            flying: false,
            config,
            fall_speed_peak: 0.0,
            in_water: false,
            was_in_water: false,
        }
    }

    /// Step the player by `dt` seconds given the current input and world.
    /// Returns whether the player is currently on the ground.
    pub fn update(&mut self, input: &mut InputState, world: &World, dt: f32) {
        // --- look ---
        let (dx, dy) = input.take_mouse_delta();
        let s = self.config.mouse_sensitivity;
        self.camera.yaw -= dx * s;
        self.camera.pitch -= dy * s;
        // Clamp pitch to avoid flipping.
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.camera.pitch = self.camera.pitch.clamp(-limit, limit);

        // --- movement intent (horizontal, relative to look) ---
        self.sneaking = input.held(Action::Sneak);
        self.sprinting =
            input.held(Action::Sprint) && input.held(Action::Forward) && !self.sneaking;

        let forward = self.camera.forward_flat();
        let right = self.camera.right();
        let mut wish = glam::Vec3::ZERO;
        if input.held(Action::Forward) {
            wish += forward;
        }
        if input.held(Action::Back) {
            wish -= forward;
        }
        if input.held(Action::Right) {
            wish += right;
        }
        if input.held(Action::Left) {
            wish -= right;
        }

        // Water detection: check if any block overlapping the player's AABB is liquid.
        let half = if self.sneaking {
            glam::Vec3::new(PLAYER_HALF.x, PLAYER_HALF_SNEAK_Y, PLAYER_HALF.z)
        } else {
            PLAYER_HALF
        };
        let aabb_min = self.pos - half;
        let aabb_max = self.pos + half;
        let min_b = world_to_block(aabb_min);
        let max_b = world_to_block(aabb_max - glam::Vec3::splat(0.001));
        self.was_in_water = self.in_water;
        let mut in_water = false;
        for by in min_b.y..=max_b.y {
            for bz in min_b.z..=max_b.z {
                for bx in min_b.x..=max_b.x {
                    if world.is_liquid(bx, by, bz) {
                        in_water = true;
                        break;
                    }
                }
                if in_water { break; }
            }
            if in_water { break; }
        }
        self.in_water = in_water;

        if self.flying {
            // Fly mode: noclip, no gravity, full 3D movement.
            let speed = self.config.fly_speed;
            if wish.length_squared() > 1e-6 {
                wish = wish.normalize() * speed;
            }
            // Vertical: Space = up, Shift = down.
            if input.held(Action::Jump) {
                wish.y = speed;
            }
            if input.held(Action::Sneak) {
                wish.y = -speed;
            }
            self.vel = wish;
            self.pos += self.vel * dt;
            self.on_ground = false;
        } else if self.in_water {
            // Water physics: buoyancy, drag, swimming.
            let swim_speed = self.config.walk_speed * 0.5;

            if wish.length_squared() > 1e-6 {
                wish = wish.normalize() * swim_speed;
            }

            self.vel.x = wish.x;
            self.vel.z = wish.z;

            // Swimming up/down: Space = swim up, Shift = swim down.
            let swim_up_speed = 3.0;
            if input.held(Action::Jump) {
                self.vel.y = swim_up_speed;
            } else if input.held(Action::Sneak) {
                self.vel.y = -swim_up_speed;
            } else {
                // Buoyancy: counteract gravity (70% buoyancy = 30% of gravity).
                let buoyant_gravity = self.config.gravity * 0.3;
                self.vel.y -= buoyant_gravity * dt;
                // Water terminal velocity (much lower than air).
                if self.vel.y < -3.0 {
                    self.vel.y = -3.0;
                }
            }

            // --- collide + move ---
            let half = if self.sneaking {
                glam::Vec3::new(PLAYER_HALF.x, PLAYER_HALF_SNEAK_Y, PLAYER_HALF.z)
            } else {
                PLAYER_HALF
            };
            let delta = self.vel * dt;
            let res = swept_aabb(world, self.pos, half, delta);
            self.pos = res.new_pos;
            // Zero velocity on blocked axes.
            if res.hit[0] || res.hit[1] {
                self.vel.x = 0.0;
            }
            if res.hit[2] || res.hit[3] {
                self.vel.y = 0.0;
            }
            if res.hit[4] || res.hit[5] {
                self.vel.z = 0.0;
            }
            self.on_ground = res.on_ground;
            self.fall_speed_peak = 0.0; // No fall damage in water.
        } else {
            // Normal physics mode.
            let speed = if self.sprinting {
                self.config.sprint_speed
            } else if self.sneaking {
                self.config.sneak_speed
            } else {
                self.config.walk_speed
            };

            if wish.length_squared() > 1e-6 {
                wish = wish.normalize() * speed;
            }

            self.vel.x = wish.x;
            self.vel.z = wish.z;

            if input.held(Action::Jump) && self.on_ground {
                self.vel.y = self.config.jump_speed;
                self.on_ground = false;
            }

            // Gravity.
            self.vel.y -= self.config.gravity * dt;
            if self.vel.y < -self.config.terminal_velocity {
                self.vel.y = -self.config.terminal_velocity;
            }

            // Track downward speed for fall damage.
            if self.vel.y < 0.0 {
                self.fall_speed_peak = self.fall_speed_peak.max(self.vel.y.abs());
            }

            // --- collide + move ---
            let half = if self.sneaking {
                glam::Vec3::new(PLAYER_HALF.x, PLAYER_HALF_SNEAK_Y, PLAYER_HALF.z)
            } else {
                PLAYER_HALF
            };
            let delta = self.vel * dt;
            let res = swept_aabb(world, self.pos, half, delta);
            self.pos = res.new_pos;
            // Zero velocity on blocked axes.
            if res.hit[0] || res.hit[1] {
                self.vel.x = 0.0;
            }
            if res.hit[2] || res.hit[3] {
                self.vel.y = 0.0;
            }
            if res.hit[4] || res.hit[5] {
                self.vel.z = 0.0;
            }
            let landed = !self.on_ground && res.on_ground;
            self.on_ground = res.on_ground;

            if landed {
                self.fall_speed_peak = 0.0;
            }
            if self.on_ground && self.vel.y <= 0.0 {
                self.fall_speed_peak = 0.0;
            }
        }

        // --- camera follows eye ---
        let eye = if self.sneaking { EYE_HEIGHT - 0.25 } else { EYE_HEIGHT };
        self.camera.pos = self.pos + glam::Vec3::new(0.0, eye, 0.0);
    }

    /// Find a safe spawn position near `x, z` by scanning down from above.
    pub fn find_spawn(world: &World, x: i32, z: i32) -> glam::Vec3 {
        for y in (1..200).rev() {
            if world.is_solid(x, y, z) {
                return glam::Vec3::new(
                    x as f32 + 0.5,
                    y as f32 + 1.0 + PLAYER_HALF.y,
                    z as f32 + 0.5,
                );
            }
        }
        glam::Vec3::new(x as f32 + 0.5, 90.0, z as f32 + 0.5)
    }

    /// True if the player's feet are currently inside a solid block (stuck).
    pub fn is_stuck(&self, world: &World) -> bool {
        let half = if self.sneaking {
            glam::Vec3::new(PLAYER_HALF.x, PLAYER_HALF_SNEAK_Y, PLAYER_HALF.z)
        } else {
            PLAYER_HALF
        };
        intersects_solid(world, self.pos, half)
    }

    pub fn eye_position(&self) -> glam::Vec3 {
        self.camera.pos
    }

    /// Look ray from the eye, for block targeting.
    pub fn look_ray(&self, reach: f32) -> voxel_core::Ray {
        voxel_core::Ray::new(self.camera.pos, self.camera.forward(), reach)
    }

    pub fn on_ground(&self) -> bool {
        self.on_ground
    }

    /// Block the player is currently standing in (feet level).
    pub fn feet_block(&self) -> glam::IVec3 {
        world_to_block(self.pos - glam::Vec3::new(0.0, PLAYER_HALF.y, 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::InputState;
    use voxel_world::world::World;
    use voxel_core::ChunkPos;
    use voxel_world::chunk::Chunk;

    fn world_with_floor() -> std::sync::Arc<World> {
        let world = World::new(42);
        let cp = ChunkPos::new(0, 0, 0);
        let mut chunk = Chunk::new(cp);
        let stone = world.registry().id_of("stone").unwrap();
        for x in 0..voxel_core::CHUNK_SIZE {
            for z in 0..voxel_core::CHUNK_SIZE {
                chunk.set(x, 0, z, stone);
            }
        }
        world.insert_chunk(cp, chunk);
        world
    }

    #[test]
    fn player_falls_and_lands() {
        let world = world_with_floor();
        let config = PlayerConfig::default();
        let mut player = Player::new(glam::Vec3::new(0.5, 5.0, 0.5), config);
        let mut input = InputState::default();
        let dt = 1.0 / 60.0;
        for _ in 0..120 {
            player.update(&mut input, &world, dt);
        }
        assert!(player.on_ground, "player should be on ground after falling");
    }

    #[test]
    fn player_jumps_off_ground() {
        let world = world_with_floor();
        let config = PlayerConfig::default();
        let mut player = Player::new(glam::Vec3::new(0.5, 1.0, 0.5), config);
        let mut input = InputState::default();
        let dt = 1.0 / 60.0;
        for _ in 0..60 {
            player.update(&mut input, &world, dt);
        }
        assert!(player.on_ground);
        input.held.insert(Action::Jump);
        player.update(&mut input, &world, dt);
        assert!(!player.on_ground, "player should leave ground when jumping");
    }

    #[test]
    fn player_mouse_look() {
        let world = world_with_floor();
        let config = PlayerConfig::default();
        let mut player = Player::new(glam::Vec3::new(0.5, 5.0, 0.5), config);
        let mut input = InputState::default();
        input.mouse_delta = (10.0, 5.0);
        player.update(&mut input, &world, 0.016);
        assert!(player.camera.yaw != 0.0);
        assert!(player.camera.pitch != 0.0);
        assert_eq!(input.mouse_delta, (0.0, 0.0));
    }

    #[test]
    fn player_pitch_clamped() {
        let world = world_with_floor();
        let config = PlayerConfig::default();
        let mut player = Player::new(glam::Vec3::new(0.5, 5.0, 0.5), config);
        let mut input = InputState::default();
        input.mouse_delta = (0.0, 1_000_000.0);
        player.update(&mut input, &world, 0.016);
        assert!(player.camera.pitch < std::f32::consts::FRAC_PI_2);
        assert!(player.camera.pitch > -std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn player_look_ray_length() {
        let config = PlayerConfig::default();
        let player = Player::new(glam::Vec3::new(0.0, 0.0, 0.0), config);
        let ray = player.look_ray(5.0);
        assert_eq!(ray.max_dist, 5.0);
    }

    #[test]
    fn player_eye_position() {
        let config = PlayerConfig::default();
        let player = Player::new(glam::Vec3::new(10.0, 20.0, 30.0), config);
        let eye = player.eye_position();
        assert!((eye.x - 10.0).abs() < 0.01);
        assert!((eye.y - 20.0 - EYE_HEIGHT).abs() < 0.01);
        assert!((eye.z - 30.0).abs() < 0.01);
    }

    #[test]
    fn is_stuck_sneaking_vs_standing() {
        let world = world_with_floor();
        let config = PlayerConfig::default();
        let mut player = Player::new(glam::Vec3::new(0.5, 2.0, 0.5), config);
        player.sneaking = false;
        assert!(!player.is_stuck(&world));
        player.sneaking = true;
        assert!(!player.is_stuck(&world));
    }
}
