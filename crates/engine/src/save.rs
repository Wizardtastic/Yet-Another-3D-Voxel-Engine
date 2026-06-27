//! Entity (player) persistence to / from `<save_dir>/entities.json`.
//!
//! World chunk persistence lives in `voxel_world::save`. This module handles
//! only the ECS-side player state (transform, velocity, AABB, input, state).

use voxel_game::Aabb;
use voxel_game::PlayerEntity;
use voxel_game::PlayerInput;
use voxel_game::PlayerState;
use voxel_game::Transform;
use voxel_game::Velocity;

/// Single source of truth for the JSON shape of a saved entity — used by
/// both `save_entities` and `load_entities` since the field set is identical.
#[derive(serde::Serialize, serde::Deserialize)]
struct EntitySave {
    name: String,
    transform: Option<Transform>,
    velocity: Option<Velocity>,
    aabb: Option<Aabb>,
    player_input: Option<PlayerInput>,
    player_state: Option<PlayerState>,
}

impl crate::EngineApp {
    /// Save all entity state (currently just the player's transform + state)
    /// to a JSON file in the save directory. The file is named
    /// `entities.json` and uses serde_json for human-readable format.
    pub(crate) fn save_entities(&self, save_dir: &std::path::Path) -> anyhow::Result<()> {
        let mut entries: Vec<EntitySave> = Vec::new();

        // Find all entities with CameraOwner (currently just the player).
        for (_entity, camera_owner) in self.ecs_world.query::<&voxel_game::CameraOwner>() {
            let _ = camera_owner;
            // We found the player. Read their components.
            let player = match self.ecs_world.resource::<PlayerEntity>().and_then(|p| p.0) {
                Some(e) => e,
                None => continue,
            };
            entries.push(EntitySave {
                name: "player".to_string(),
                transform: self.ecs_world.get::<Transform>(player).copied(),
                velocity: self.ecs_world.get::<Velocity>(player).copied(),
                aabb: self.ecs_world.get::<Aabb>(player).copied(),
                player_input: self.ecs_world.get::<PlayerInput>(player).copied(),
                player_state: self.ecs_world.get::<PlayerState>(player).copied(),
            });
            break;
        }

        let json = serde_json::to_string_pretty(&entries)?;
        std::fs::write(save_dir.join("entities.json"), json)?;
        Ok(())
    }

    /// Load entity state from a JSON file in the save directory and restore
    /// components on the existing player entity. A missing file is treated
    /// as "no entities to load" (not an error).
    pub(crate) fn load_entities(&mut self, save_dir: &std::path::Path) -> anyhow::Result<()> {
        let path = save_dir.join("entities.json");
        let json = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => return Ok(()), // no entities file = no entities to load
        };
        let entries: Vec<EntitySave> = serde_json::from_str(&json)?;

        let player = match self.ecs_world.resource::<PlayerEntity>().and_then(|p| p.0) {
            Some(e) => e,
            None => return Ok(()),
        };

        for entry in entries {
            if entry.name != "player" {
                continue;
            }
            if let Some(t) = entry.transform {
                self.ecs_world.set(player, t);
            }
            if let Some(v) = entry.velocity {
                self.ecs_world.set(player, v);
            }
            if let Some(a) = entry.aabb {
                self.ecs_world.set(player, a);
            }
            if let Some(pi) = entry.player_input {
                self.ecs_world.set(player, pi);
            }
            if let Some(ps) = entry.player_state {
                self.ecs_world.set(player, ps);
            }
            break;
        }
        Ok(())
    }
}
