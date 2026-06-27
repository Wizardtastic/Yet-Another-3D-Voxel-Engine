//! Chat command dispatcher.
//!
//! The `execute_command` method on [`crate::EngineApp`] is the single
//! destination for `/`-prefixed chat commands. The match arms live here
//! in isolation so the rest of `lib.rs` doesn't have to scroll through
//! ~280 lines of dispatcher boilerplate.

use voxel_game::CommandResult;

impl crate::EngineApp {
    /// Execute a chat command.
    pub(crate) fn execute_command(&mut self, result: CommandResult) {
        match result {
            CommandResult::Teleport(target) => {
                // Legacy stub.
                self.gameplay.player.pos = target;
                self.gameplay.player.camera.pos =
                    target + glam::Vec3::new(0.0, voxel_game::player::EYE_HEIGHT, 0.0);
                self.gameplay.player.vel = glam::Vec3::ZERO;
                // ECS source of truth.
                crate::GamePlayState::set_player_pos(&mut self.ecs_world, target);
                // Also update the cached camera resource so the next frame
                // doesn't snap back to the old position.
                if let Some(cam) = self.ecs_world.resource_mut::<voxel_game::CameraResource>() {
                    cam.0.pos = target + glam::Vec3::new(0.0, voxel_game::EYE_HEIGHT, 0.0);
                }
                self.gameplay
                    .chat
                    .push_message(format!(
                        "Teleported to ({:.1}, {:.1}, {:.1})",
                        target.x, target.y, target.z
                    ));
            }
            CommandResult::SetTime(t) => {
                self.gameplay.game_time = t % self.gameplay.day_length;
                self.gameplay
                    .chat
                    .push_message(format!("Time set to {:.0}s", self.gameplay.game_time));
            }
            CommandResult::TimeSpeed(mult) => {
                if mult <= 0.0 {
                    self.gameplay.chat.push_message("time speed must be positive".into());
                    return;
                }
                self.gameplay.day_length = self.config.day_length / mult;
                self.gameplay.chat.push_message(format!(
                    "Day length: {:.0}s ({:.1}x speed)",
                    self.gameplay.day_length, mult
                ));
            }
            CommandResult::Give(block, count) => {
                self.gameplay
                    .chat
                    .push_message(format!("Gave {count} {block} (not yet implemented)"));
            }
            CommandResult::SetBlock(x, y, z, block) => {
                let reg = self.world_state.world.registry();
                if let Some(id) = reg.id_of(&block) {
                    let old = self.world_state.world.get_block(x, y, z);
                    self.world_state.world.set_block(x, y, z, id);
                    self.gameplay.undo_redo.push(voxel_game::EditAction {
                        edits: vec![voxel_game::BlockEdit {
                            x,
                            y,
                            z,
                            old_block: old.0,
                            new_block: id.0,
                        }],
                    });
                    self.gameplay
                        .chat
                        .push_message(format!("Block set at ({x}, {y}, {z})"));
                } else {
                    self.gameplay.chat.push_message(format!("Unknown block: {block}"));
                }
            }
            CommandResult::Fill(x1, y1, z1, x2, y2, z2, block) => {
                let reg = self.world_state.world.registry();
                if let Some(id) = reg.id_of(&block) {
                    let min_x = x1.min(x2);
                    let max_x = x1.max(x2);
                    let min_y = y1.min(y2);
                    let max_y = y1.max(y2);
                    let min_z = z1.min(z2);
                    let max_z = z1.max(z2);
                    let mut count = 0u32;
                    let mut edits = Vec::new();
                    for x in min_x..=max_x {
                        for y in min_y..=max_y {
                            for z in min_z..=max_z {
                                let old = self.world_state.world.get_block(x, y, z);
                                self.world_state.world.set_block(x, y, z, id);
                                edits.push(voxel_game::BlockEdit {
                                    x,
                                    y,
                                    z,
                                    old_block: old.0,
                                    new_block: id.0,
                                });
                                count += 1;
                            }
                        }
                    }
                    self.gameplay.undo_redo.push(voxel_game::EditAction { edits });
                    self.gameplay.chat.push_message(format!("Filled {count} blocks"));
                } else {
                    self.gameplay.chat.push_message(format!("Unknown block: {block}"));
                }
            }
            CommandResult::Gamemode(_mode) => {
                self.gameplay
                    .chat
                    .push_message("Gamemode not yet implemented".into());
            }
            CommandResult::Position => {
                let p = crate::GamePlayState::player_pos(&self.ecs_world)
                    .unwrap_or(self.gameplay.player.pos);
                self.gameplay
                    .chat
                    .push_message(format!("Pos: ({:.1}, {:.1}, {:.1})", p.x, p.y, p.z));
            }
            CommandResult::ChunkInfo => {
                let p = crate::GamePlayState::player_pos(&self.ecs_world)
                    .unwrap_or(self.gameplay.player.pos);
                let block = voxel_core::math::world_to_block(p);
                let cp = voxel_core::math::block_to_chunk(block);
                let loaded = self.world_state.world.loaded_chunk_count();
                let meshed = self.world_state.world.meshed_chunk_count();
                self.gameplay.chat.push_message(format!(
                    "Chunk ({}, {}), loaded: {loaded}, meshed: {meshed}",
                    cp.x(),
                    cp.z()
                ));
            }
            CommandResult::Fps => {
                self.gameplay.chat.push_message(format!(
                    "Frame time: {:.1}ms ({:.0} fps)",
                    self.input.frame_time * 1000.0,
                    1.0 / self.input.frame_time.max(0.001)
                ));
            }
            CommandResult::Reload => {
                self.gameplay
                    .chat
                    .push_message("Config reloaded (not yet implemented)".into());
            }
            CommandResult::Clear => {
                self.gameplay.chat.messages.clear();
                self.gameplay.chat.push_message("Chat cleared".into());
            }
            CommandResult::Save(path) => {
                let save_dir = std::path::PathBuf::from(&path);
                match voxel_world::save::save_world(&self.world_state.world, &save_dir) {
                    Ok(()) => {
                        if let Err(e) = self.save_entities(&save_dir) {
                            self.gameplay
                                .chat
                                .push_message(format!("World saved but entity save failed: {e}"));
                        } else {
                            self.gameplay.chat.push_message(format!("World saved to {path}"));
                        }
                    }
                    Err(e) => self.gameplay.chat.push_message(format!("Save failed: {e}")),
                }
            }
            CommandResult::Load(path) => {
                let save_dir = std::path::PathBuf::from(&path);
                match voxel_world::save::load_world(&save_dir) {
                    Ok((seed, chunks)) => {
                        self.config.seed = seed;
                        let count = chunks.len();
                        self.world_state.world.insert_chunks(chunks);
                        if let Err(e) = self.load_entities(&save_dir) {
                            self.gameplay.chat.push_message(format!(
                                "Loaded {count} chunks but entity load failed: {e}"
                            ));
                        } else {
                            self.gameplay
                                .chat
                                .push_message(format!("Loaded {count} chunks from {path}"));
                        }
                        let pos = crate::GamePlayState::player_pos(&self.ecs_world)
                            .unwrap_or(self.gameplay.player.pos);
                        if let Some(s) = &self.world_state.streamer {
                            s.set_focus(pos);
                        }
                    }
                    Err(e) => self.gameplay.chat.push_message(format!("Load failed: {e}")),
                }
            }
            CommandResult::Copy(x1, y1, z1, x2, y2, z2) => {
                let min_x = x1.min(x2);
                let max_x = x1.max(x2);
                let min_y = y1.min(y2);
                let max_y = y1.max(y2);
                let min_z = z1.min(z2);
                let max_z = z1.max(z2);
                let mut blocks = Vec::new();
                for x in min_x..=max_x {
                    for y in min_y..=max_y {
                        for z in min_z..=max_z {
                            blocks.push(self.world_state.world.get_block(x, y, z));
                        }
                    }
                }
                let count = blocks.len();
                self.gameplay.clipboard =
                    Some(((min_x, min_y, min_z), (max_x, max_y, max_z), blocks));
                let size = (max_x - min_x + 1) * (max_y - min_y + 1) * (max_z - min_z + 1);
                self.gameplay
                    .chat
                    .push_message(format!("Copied {size} blocks ({count} stored)"));
            }
            CommandResult::Paste => {
                let Some(((min_x, min_y, min_z), (max_x, max_y, max_z), blocks)) =
                    &self.gameplay.clipboard
                else {
                    self.gameplay.chat.push_message("Clipboard empty".into());
                    return;
                };
                let sx = max_x - min_x + 1;
                let sy = max_y - min_y + 1;
                let mut count = 0u32;
                let mut idx = 0;
                let mut edits = Vec::new();
                for x in 0..sx {
                    for y in 0..sy {
                        for z in 0..(max_z - min_z + 1) {
                            let id = blocks[idx];
                            let old =
                                self.world_state.world.get_block(min_x + x, min_y + y, min_z + z);
                            self.world_state
                                .world
                                .set_block(min_x + x, min_y + y, min_z + z, id);
                            edits.push(voxel_game::BlockEdit {
                                x: min_x + x,
                                y: min_y + y,
                                z: min_z + z,
                                old_block: old.0,
                                new_block: id.0,
                            });
                            count += 1;
                            idx += 1;
                        }
                    }
                }
                self.gameplay.undo_redo.push(voxel_game::EditAction { edits });
                self.gameplay.chat.push_message(format!("Pasted {count} blocks"));
            }
            CommandResult::Help => {
                self.gameplay.chat.push_message("Commands:".into());
                self.gameplay
                    .chat
                    .push_message("  /tp x y z        - teleport (~ for relative)".into());
                self.gameplay.chat.push_message(
                    "  /time set <val>  - set time (day/night/dawn/dusk/seconds)".into(),
                );
                self.gameplay
                    .chat
                    .push_message("  /time speed <x>  - set time speed multiplier".into());
                self.gameplay
                    .chat
                    .push_message("  /give <block> [n]- give block items (WIP)".into());
                self.gameplay.chat.push_message("  /setblock x y z <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /fill x1 y1 z1 x2 y2 z2 <block>".into());
                self.gameplay
                    .chat
                    .push_message("  /gamemode <mode> - set gamemode (WIP)".into());
                self.gameplay
                    .chat
                    .push_message("  /pos             - show current position".into());
                self.gameplay
                    .chat
                    .push_message("  /chunk           - show chunk info".into());
                self.gameplay
                    .chat
                    .push_message("  /fps             - show frame rate".into());
                self.gameplay
                    .chat
                    .push_message("  /clear           - clear chat".into());
                self.gameplay
                    .chat
                    .push_message("  /save [path]     - save world to disk".into());
                self.gameplay
                    .chat
                    .push_message("  /load [path]     - load world from disk".into());
                self.gameplay
                    .chat
                    .push_message("  /copy x1 y1 z1 x2 y2 z2 - copy region".into());
                self.gameplay
                    .chat
                    .push_message("  /paste           - paste clipboard".into());
                self.gameplay
                    .chat
                    .push_message("  /help            - show this help".into());
                self.gameplay
                    .chat
                    .push_message("  Tab = autocomplete, Up/Down = history".into());
            }
            CommandResult::Empty => {}
            CommandResult::Unknown(msg) => {
                self.gameplay.chat.push_message(msg);
            }
        }
    }
}
