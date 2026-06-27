//! Per-frame engine orchestration: drain the chunk streamer, run the
//! fixed-timestep ECS schedule, update camera + day/night sky, render,
//! handle auto/screenshot capture.

use std::time::Instant;

use voxel_render::UiDrawData;
use voxel_world::ChunkStreamEvent;

/// Fixed-timestep simulation step used by both the ECS schedule and any
/// standalone physics updates inside the frame loop.
const FIXED_DT: f64 = 1.0 / 60.0;

impl crate::EngineApp {
    /// Build the ChunkUpload list from a stream event's mesh bundle.
    /// Each upload owns its vertex/index bytes via `to_vec()` on a slice
    /// cast from the bundle's POD arrays; this replaces the previous
    /// scratch-then-clone pattern that did two full copies per chunk with
    /// a single allocation. Only called from `frame()` in this module.
    fn uploads_from_bundle(
        pos: voxel_core::ChunkPos,
        bundle: &voxel_world::ChunkMeshBundle,
    ) -> Vec<voxel_render::ChunkUpload> {
        let mut out = Vec::new();
        if !bundle.opaque.is_empty() {
            out.push(voxel_render::ChunkUpload {
                pos,
                pass: voxel_render::MeshPass::Opaque,
                vertices: bytemuck::cast_slice(&bundle.opaque.vertices).to_vec(),
                indices: bytemuck::cast_slice(&bundle.opaque.indices).to_vec(),
                index_count: bundle.opaque.indices.len() as u32,
            });
        }
        if !bundle.transparent.is_empty() {
            out.push(voxel_render::ChunkUpload {
                pos,
                pass: voxel_render::MeshPass::Transparent,
                vertices: bytemuck::cast_slice(&bundle.transparent.vertices).to_vec(),
                indices: bytemuck::cast_slice(&bundle.transparent.indices).to_vec(),
                index_count: bundle.transparent.indices.len() as u32,
            });
        }
        out
    }

    /// Run one render frame: drain streamer events, upload meshes, run the
    /// fixed-timestep sim, then render + present.
    pub(crate) fn frame(&mut self) {
        // Drain streamer events and sync GPU buffers.
        if let Some(streamer) = &self.world_state.streamer {
            let events = streamer.poll_events();
            let mut uploads = Vec::new();
            for ev in events {
                match ev {
                    ChunkStreamEvent::MeshReady { pos, bundle } => {
                        uploads.extend(Self::uploads_from_bundle(pos, &bundle));
                    }
                    ChunkStreamEvent::Unloaded(pos) => {
                        if let Some(r) = self.render.renderer.as_mut() {
                            r.remove_chunk(pos);
                        }
                    }
                    ChunkStreamEvent::Generated(_) => {}
                }
            }
            if !uploads.is_empty() {
                if let Some(r) = self.render.renderer.as_mut() {
                    r.upload_chunks(uploads);
                }
            }
        }

        // Fixed-timestep simulation.
        let now = Instant::now();
        let frame_dt = now.duration_since(self.input.last_time).as_secs_f64();
        self.input.last_time = now;
        self.input.frame_time = frame_dt;
        self.input.sim_accumulator += frame_dt;
        // Clamp to avoid spiral-of-death after long pauses.
        if self.input.sim_accumulator > 0.25 {
            self.input.sim_accumulator = 0.25;
        }

        // Wait for the player's chunk to load before running physics, so the
        // player doesn't fall through unloaded terrain. Once the first chunk
        // arrives, snap the player to the surface.
        if !self.input.spawned {
            let p = crate::GamePlayState::player_pos(&self.ecs_world)
                .unwrap_or(self.gameplay.player.pos);
            if self.world_state.world.is_block_loaded(
                p.x.floor() as i32,
                p.y.floor() as i32,
                p.z.floor() as i32,
            ) {
                // Find the surface: scan down from current Y for the first solid.
                let mut surface_y = p.y.floor() as i32;
                for y in ((p.y.floor() as i32 - 20).max(1)..=p.y.floor() as i32 + 5).rev() {
                    if self
                        .world_state
                        .world
                        .is_solid(p.x.floor() as i32, y, p.z.floor() as i32)
                    {
                        surface_y = y;
                        break;
                    }
                }
                // Place the player standing on the surface.
                let new_pos = glam::Vec3::new(
                    p.x,
                    surface_y as f32 + 1.0 + voxel_game::player::PLAYER_HALF.y,
                    p.z,
                );
                // Legacy stub mirror.
                self.gameplay.player.pos = new_pos;
                self.gameplay.player.vel.y = 0.0;
                self.gameplay.player.camera.pos = self.gameplay.player.pos
                    + glam::Vec3::new(0.0, voxel_game::player::EYE_HEIGHT, 0.0);
                // ECS source of truth.
                crate::GamePlayState::set_player_pos(&mut self.ecs_world, new_pos);
                self.input.spawned = true;
                log::info!(
                    "spawn ready: player at ({:.1}, {:.1}, {:.1}), surface_y={}",
                    self.gameplay.player.pos.x,
                    self.gameplay.player.pos.y,
                    self.gameplay.player.pos.z,
                    surface_y
                );
            } else {
                // Chunk not loaded yet — skip physics, still render.
                self.input.sim_accumulator = 0.0;
            }
        }

        // Consume clicks once per frame.
        let mut clicks = self.input.input.take_clicks();

        // Handle block picker clicks.
        if self.gameplay.block_picker_open && clicks.left {
            self.handle_block_picker_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Handle pause-menu clicks (only in PauseMenu state).
        if self.gameplay.game_state == crate::GameState::PauseMenu && clicks.left {
            self.handle_pause_click();
            clicks.left = false;
            clicks.right = false;
        }

        // Run simulation only when playing (not paused). The fixed-step
        // schedule runs ECS systems in input -> movement -> lifecycle order.
        if self.gameplay.game_state == crate::GameState::Playing {
            // Project the engine's resolved input into the per-step snapshot
            // the systems expect.
            let snap = voxel_game::InputSnapshot {
                forward: self.input.input.held(voxel_game::input::Action::Forward),
                back: self.input.input.held(voxel_game::input::Action::Back),
                left: self.input.input.held(voxel_game::input::Action::Left),
                right: self.input.input.held(voxel_game::input::Action::Right),
                jump: self.input.input.held(voxel_game::input::Action::Jump),
                sneak: self.input.input.held(voxel_game::input::Action::Sneak),
                sprint: self.input.input.held(voxel_game::input::Action::Sprint),
                flying: crate::GamePlayState::player_flying(&self.ecs_world),
                mouse_delta: self.input.input.mouse_delta,
            };
            // The input_system will copy this into PlayerInput, so reset the
            // engine's delta after projection to avoid double-applying it on
            // the next frame.
            self.input.input.mouse_delta = (0.0, 0.0);
            self.ecs_world
                .insert_resource(voxel_game::InputResource(snap));
            self.ecs_world
                .insert_resource(voxel_game::PhysicsWorldRes(self.world_state.world.clone()));

            let mut steps = 0;
            while self.input.sim_accumulator >= FIXED_DT && steps < 8 {
                if let Some(sched) = self.schedule.as_mut() {
                    sched.run(&mut self.ecs_world, FIXED_DT as f32);
                }
                self.input.sim_accumulator -= FIXED_DT;
                steps += 1;
            }
        }

        // Mirror ECS state back into the legacy `Player` stub so any code
        // path that still reads it sees the latest state.
        if let Some(pos) = crate::GamePlayState::player_pos(&self.ecs_world) {
            self.gameplay.player.pos = pos;
        }
        if let Some(vel) = crate::GamePlayState::player_vel(&self.ecs_world) {
            self.gameplay.player.vel = vel;
        }
        self.gameplay.player.flying = crate::GamePlayState::player_flying(&self.ecs_world);

        // Refresh the cached camera resource from the player's transform +
        // current eye offset. This is the camera the renderer will use.
        let player_camera_input: Option<(voxel_game::Transform, voxel_game::PlayerState)> =
            crate::GamePlayState::player_entity(&self.ecs_world).and_then(|entity| {
                let t = self.ecs_world.get::<voxel_game::Transform>(entity)?.clone();
                let s = self.ecs_world.get::<voxel_game::PlayerState>(entity)?.clone();
                Some((t, s))
            });
        if let (Some((t, s)), Some(cam_res)) = (
            player_camera_input,
            self.ecs_world.resource_mut::<voxel_game::CameraResource>(),
        ) {
            voxel_game::update_camera_from_transform(&mut cam_res.0, &t, &s);
        }

        // Drive incremental water flow.
        let water_affected = if self.gameplay.game_state == crate::GameState::Playing {
            self.world_state.world.tick_water(frame_dt as f32)
        } else {
            std::collections::HashSet::new()
        };
        if !water_affected.is_empty() {
            if let Some(streamer) = &self.world_state.streamer {
                for cp in water_affected {
                    streamer.request_remesh(cp);
                }
            }
        }

        // Keep the streamer centred on the player (even before spawn, so the
        // spawn chunk loads ASAP).
        let player_pos_now = crate::GamePlayState::player_pos(&self.ecs_world)
            .unwrap_or(self.gameplay.player.pos);
        let camera_now = crate::GamePlayState::player_camera(&self.ecs_world)
            .unwrap_or(self.gameplay.player.camera);
        if let Some(streamer) = &self.world_state.streamer {
            // Only send focus if player moved to a different chunk.
            let player_chunk =
                voxel_core::math::block_to_chunk(voxel_core::math::world_to_block(player_pos_now));
            let player_chunk_v = voxel_core::math::chunk_origin(player_chunk).as_vec3();
            if (player_chunk_v - self.input.last_focus_pos).length_squared() > 1.0 {
                streamer.set_focus(player_pos_now);
                self.input.last_focus_pos = player_chunk_v;
            }
            // Only send sun_dir if it changed.
            let dp = self.day_params();
            let sun_dir = glam::Vec3::new(
                dp.sun_angle.cos() * 0.3,
                dp.sun_altitude,
                dp.sun_angle.sin() * 0.3,
            )
            .normalize();
            if (sun_dir - self.input.last_sun_dir).length_squared() > 0.0001 {
                streamer.set_sun_dir(sun_dir);
                self.input.last_sun_dir = sun_dir;
            }
            // Send frustum for mesh-building culling.
            let mut cam = camera_now;
            let h = self.render.window_size.1 as f32;
            cam.aspect = if h > 0.0 {
                self.render.window_size.0 as f32 / h
            } else {
                1.0
            };
            let vp = cam.view_projection();
            let frustum = voxel_core::Frustum::from_view_projection(vp);
            streamer.set_frustum(frustum);
        }

        // Block interactions only when playing, chat closed, picker closed.
        if (clicks.left || clicks.right)
            && self.input.cursor_locked
            && self.gameplay.game_state == crate::GameState::Playing
            && !self.gameplay.chat.open
            && !self.gameplay.block_picker_open
        {
            if let Some(streamer) = &self.world_state.streamer {
                let eye = crate::GamePlayState::player_eye_pos(&self.ecs_world)
                    .unwrap_or(camera_now.pos);
                let result = voxel_game::BlockAction::apply(
                    &self.world_state.world,
                    streamer,
                    &mut self.gameplay.hotbar,
                    eye,
                    camera_now.forward(),
                    clicks,
                    player_pos_now,
                );
                if !result.edits.is_empty() {
                    self.gameplay.undo_redo.push(voxel_game::EditAction {
                        edits: result.edits,
                    });
                }
            }
        }

        // Advance game time (day/night cycle). Modulo so a long frame wraps
        // correctly even if frame_dt > day_length.
        if self.gameplay.game_state == crate::GameState::Playing {
            self.gameplay.game_time =
                (self.gameplay.game_time + frame_dt) % self.gameplay.day_length;
        }

        // Build UI overlay.
        let ui: UiDrawData = self.build_ui();

        // Update sky parameters for day/night.
        let dp = self.day_params();
        let sun_dir = [
            dp.sun_angle.cos() * 0.3,
            dp.sun_altitude,
            dp.sun_angle.sin() * 0.3,
        ];
        self.world_state.world.set_sun_dir(
            glam::Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]).normalize(),
        );

        // Detect if camera is underwater for visual effects.
        let eye_block = voxel_core::math::world_to_block(camera_now.pos);
        let underwater =
            self.world_state
                .world
                .is_liquid(eye_block.x, eye_block.y, eye_block.z);

        // Render.
        let mut camera = camera_now;
        let h = self.render.window_size.1 as f32;
        camera.aspect = if h > 0.0 {
            self.render.window_size.0 as f32 / h
        } else {
            1.0
        };
        if let Some(r) = self.render.renderer.as_mut() {
            r.set_sky(
                dp.horizon,
                dp.zenith,
                dp.fog,
                dp.daylight.max(0.15),
                underwater,
            );
            r.set_sun_dir(sun_dir);
            if self.config.shadow_enabled {
                let (cascade_vps, cascade_splits, light_dir_and_bias) = compute_shadow_cascades(
                    &camera,
                    glam::Vec3::new(sun_dir[0], sun_dir[1], sun_dir[2]),
                    0.1,
                    self.config.render.fog_distance,
                );
                r.set_shadow_data(cascade_vps, cascade_splits, light_dir_and_bias);
            }
            r.set_post_params(
                self.config.exposure,
                self.config.vignette_strength,
                self.gameplay.game_time as f32,
            );
            if let Err(e) =
                r.draw_frame(camera, Some(&ui), self.gameplay.game_time as f32, underwater)
            {
                log::error!("draw_frame: {e}");
            }
            // Collect profiler data.
            if self.profiler.enabled {
                self.profiler.end_frame(self.input.frame_time * 1000.0);
                let gpu = r.latest_timings();
                self.profiler.gpu_timings.push_back(gpu);
                if self.profiler.gpu_timings.len() > 120 {
                    self.profiler.gpu_timings.pop_front();
                }
            }
        }

        // Screenshot requested via keybind (deferred from event handler).
        if self.input.screenshot_requested {
            self.input.screenshot_requested = false;
            self.do_capture();
        }

        // Auto-capture for verification.
        self.input.frame_count += 1;
        if let Some(after) = self.config.capture_after_frames {
            if !self.input.captured && self.input.frame_count >= after {
                self.input.captured = true;
                self.do_capture();
            }
        }
    }

    /// Render a still frame to PNG via the auto-capture machinery.
    pub(crate) fn do_capture(&mut self) {
        let mut camera = crate::GamePlayState::player_camera(&self.ecs_world)
            .unwrap_or(self.gameplay.player.camera);
        let h = self.render.window_size.1 as f32;
        camera.aspect = if h > 0.0 {
            self.render.window_size.0 as f32 / h
        } else {
            1.0
        };
        let ui = self.build_ui();
        let dp = self.day_params();
        let eye_block = voxel_core::math::world_to_block(camera.pos);
        let underwater =
            self.world_state
                .world
                .is_liquid(eye_block.x, eye_block.y, eye_block.z);
        let Some(r) = self.render.renderer.as_mut() else {
            return;
        };
        r.set_sky(
            dp.horizon,
            dp.zenith,
            dp.fog,
            dp.daylight.max(0.15),
            underwater,
        );
        match r.capture_frame(camera, Some(&ui), self.gameplay.game_time as f32, underwater) {
            Ok(rgba) => {
                let (w, h) = self.render.window_size;
                if let Some(img) = image::RgbaImage::from_raw(w, h, rgba) {
                    match img.save(&self.config.capture_path) {
                        Ok(()) => log::info!(
                            "captured frame to {} ({}x{}, chunks={})",
                            self.config.capture_path.display(),
                            w,
                            h,
                            r.chunk_count()
                        ),
                        Err(e) => log::error!("save capture: {e}"),
                    }
                } else {
                    log::error!("capture: image size mismatch, failed to create RgbaImage");
                }
            }
            Err(e) => log::error!("capture_frame: {e}"),
        }
    }
}

/// Compute cascaded shadow-map view-projection matrices for the four cascades
/// used by the renderer. Each cascade projects the camera frustum sub-frustum
/// from the sun's POV.
fn compute_shadow_cascades(
    camera: &voxel_core::Camera,
    sun_dir: glam::Vec3,
    near: f32,
    far: f32,
) -> ([[f32; 16]; 4], [f32; 4], [f32; 4]) {
    let split_factors = [0.05, 0.15, 0.4, 1.0];
    let mut cascade_vps = [[0.0f32; 16]; 4];
    let mut cascade_splits = [0.0f32; 4];

    let view_proj = camera.view_projection();
    let inv_vp = view_proj.inverse();

    let sun_dir = sun_dir.normalize();

    for i in 0..4 {
        let prev_split = if i == 0 { 0.0 } else { split_factors[i - 1] };
        let split = split_factors[i];

        let near_split = near + (far - near) * prev_split;
        let far_split = near + (far - near) * split;
        cascade_splits[i] = far_split;

        let corners_ndc = [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ];

        let mut corners_world = [glam::Vec3::ZERO; 8];
        for j in 0..8 {
            let ndc_z = if j < 4 {
                -1.0 + 2.0 * (near_split - near) / (far - near)
            } else {
                -1.0 + 2.0 * (far_split - near) / (far - near)
            };
            let ndc = glam::Vec4::new(corners_ndc[j][0], corners_ndc[j][1], ndc_z, 1.0);
            let world = inv_vp * ndc;
            corners_world[j] = world.truncate() / world.w;
        }

        let center = corners_world.iter().fold(glam::Vec3::ZERO, |acc, &c| acc + c) / 8.0;

        let mut min = corners_world[0];
        let mut max = corners_world[0];
        for &c in &corners_world[1..] {
            min = min.min(c);
            max = max.max(c);
        }

        let light_pos = center - sun_dir * 100.0;
        let up = if sun_dir.y.abs() > 0.99 {
            glam::Vec3::new(1.0, 0.0, 0.0)
        } else {
            glam::Vec3::new(0.0, 1.0, 0.0)
        };
        let light_view = glam::Mat4::look_at_lh(light_pos, center, up);

        let mut light_min = glam::Vec3::ZERO;
        let mut light_max = glam::Vec3::ZERO;
        for &c in &corners_world {
            let lc = light_view * glam::Vec4::new(c.x, c.y, c.z, 1.0);
            let lc = lc.truncate();
            if lc.x < light_min.x {
                light_min.x = lc.x;
            }
            if lc.x > light_max.x {
                light_max.x = lc.x;
            }
            if lc.y < light_min.y {
                light_min.y = lc.y;
            }
            if lc.y > light_max.y {
                light_max.y = lc.y;
            }
            if lc.z < light_min.z {
                light_min.z = lc.z;
            }
            if lc.z > light_max.z {
                light_max.z = lc.z;
            }
        }

        let pad = 2.0;
        light_min -= glam::Vec3::splat(pad);
        light_max += glam::Vec3::splat(pad);

        let radius = (light_max - light_min).length() * 0.5;

        let light_proj = glam::Mat4::orthographic_lh(
            light_min.x,
            light_max.x,
            light_min.y,
            light_max.y,
            -radius - 50.0,
            radius + 50.0,
        );

        let vp = light_proj * light_view;
        cascade_vps[i] = vp.to_cols_array();
    }

    let light_dir_and_bias = [sun_dir.x, sun_dir.y, sun_dir.z, 0.01];

    (cascade_vps, cascade_splits, light_dir_and_bias)
}
