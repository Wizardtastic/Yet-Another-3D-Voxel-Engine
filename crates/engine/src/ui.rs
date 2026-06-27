//! UI overlay rendering and click-handling.
//!
//! All `draw_*` methods on `EngineApp` plus their paired click handlers
//! (`handle_*_click`) live here. The split keeps the frame loop in
//! `lib.rs` focused on per-frame orchestration rather than HUD layout.

use voxel_render::UiDrawData;

use crate::{GamePlayState, GameState};

impl crate::EngineApp {
    /// Build the UI overlay for this frame: crosshair + hotbar when playing,
    /// or the pause/exit menu when paused.
    pub(crate) fn build_ui(&mut self) -> UiDrawData {
        let mut ui = UiDrawData::default();
        let (w, h) = self.render.window_size;

        if self.gameplay.game_state == GameState::Playing {
            self.draw_crosshair(&mut ui, w as f32, h as f32);
            self.draw_hotbar(&mut ui, w as f32, h as f32);
            if self.gameplay.debug_overlay {
                self.draw_debug_overlay(&mut ui, w as f32, h as f32);
            }
            if self.profiler.enabled {
                self.draw_profiler_overlay(&mut ui, w as f32, h as f32);
            }
            if self.gameplay.block_picker_open {
                self.draw_block_picker(&mut ui, w as f32, h as f32);
            }
            self.draw_chat(&mut ui, w as f32, h as f32);
        } else {
            self.draw_pause_menu(&mut ui, w as f32, h as f32);
        }

        ui
    }

    /// Draw a centred crosshair (two thin white bars forming a +).
    fn draw_crosshair(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let cx = w * 0.5;
        let cy = h * 0.5;
        let len = 10.0;
        let thick = 2.0;
        // Blend for slight transparency so it's visible on any background.
        let color = [255, 255, 255, 200];
        ui.quad(cx - len, cy - thick * 0.5, len * 2.0, thick, color);
        ui.quad(cx - thick * 0.5, cy - len, thick, len * 2.0, color);
    }

    /// Draw the 9-slot hotbar at the bottom-centre of the screen.
    fn draw_hotbar(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let slot = 48.0;
        let gap = 4.0;
        let total = slot * 9.0 + gap * 8.0;
        let x0 = (w - total) * 0.5;
        let y0 = h - slot - 12.0;
        let reg = self.world_state.world.registry();

        for i in 0..9 {
            let x = x0 + i as f32 * (slot + gap);
            // Slot background (dark semi-transparent).
            ui.quad(x, y0, slot, slot, [40, 40, 40, 180]);
            // Slot border.
            ui.rect_border(x, y0, slot, slot, 2.0, [80, 80, 80, 220]);

            // Block icon.
            let block_id = self.gameplay.hotbar.slot(i);
            if !block_id.is_air() {
                let def = reg.get(block_id);
                let tile = def.textures.tile(voxel_world::registry::Face::PosX);
                let icon_size = slot - 8.0;
                ui.block_icon(
                    x + 4.0,
                    y0 + 4.0,
                    icon_size,
                    icon_size,
                    tile,
                    [255, 255, 255, 255],
                );
            }

            // Selection highlight on the current slot.
            if i == self.gameplay.hotbar.selected {
                ui.rect_border(
                    x - 2.0,
                    y0 - 2.0,
                    slot + 4.0,
                    slot + 4.0,
                    3.0,
                    [255, 255, 255, 255],
                );
            }
        }
    }

    /// Draw the pause/exit menu: dark overlay, centred panel, title, two buttons.
    fn draw_pause_menu(&mut self, ui: &mut UiDrawData, w: f32, h: f32) {
        // Full-screen dark overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 160]);

        let panel_w = 320.0;
        let panel_h = 220.0;
        let px = (w - panel_w) * 0.5;
        let py = (h - panel_h) * 0.5;

        // Panel background.
        ui.quad(px, py, panel_w, panel_h, [30, 30, 40, 240]);
        ui.rect_border(px, py, panel_w, panel_h, 2.0, [100, 100, 120, 255]);

        // Title "PAUSED".
        let title = "PAUSED";
        let tw = self.render.font.text_width(title, 2.0);
        ui.text(
            title,
            px + (panel_w - tw) * 0.5,
            py + 20.0,
            2.0,
            [255, 255, 255, 255],
            &self.render.font,
        );

        // Buttons.
        let btn_w = 240.0;
        let btn_h = 44.0;
        let btn_x = px + (panel_w - btn_w) * 0.5;
        let btn_y0 = py + 70.0;
        let btn_y1 = py + 130.0;

        // Back to Game button.
        self.draw_button(
            ui,
            btn_x,
            btn_y0,
            btn_w,
            btn_h,
            "BACK TO GAME",
            [40, 80, 40, 220],
            [60, 120, 60, 255],
            [120, 200, 120, 255],
        );

        // Exit Game button.
        self.draw_button(
            ui,
            btn_x,
            btn_y1,
            btn_w,
            btn_h,
            "EXIT GAME",
            [90, 30, 30, 220],
            [140, 50, 50, 255],
            [220, 100, 100, 255],
        );

        self.gameplay.pause_buttons =
            Some([(btn_x, btn_y0, btn_w, btn_h), (btn_x, btn_y1, btn_w, btn_h)]);
    }

    /// Check if a point is inside a rectangle.
    fn point_in_rect(&self, pos: (f32, f32), x: f32, y: f32, w: f32, h: f32) -> bool {
        pos.0 >= x && pos.0 <= x + w && pos.1 >= y && pos.1 <= y + h
    }

    /// Draw a hover-able button (filled quad + 2px border + centred label).
    /// `fill_normal` is used when the cursor isn't over the rect; `fill_hover`
    /// when it is.
    fn draw_button(
        &self,
        ui: &mut UiDrawData,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        label: &str,
        fill_normal: [u8; 4],
        fill_hover: [u8; 4],
        border: [u8; 4],
    ) {
        let hovered = self.point_in_rect(self.gameplay.mouse_pos, x, y, w, h);
        let fill = if hovered { fill_hover } else { fill_normal };
        ui.quad(x, y, w, h, fill);
        ui.rect_border(x, y, w, h, 2.0, border);
        let scale = 1.5;
        let lw = self.render.font.text_width(label, scale);
        ui.text(
            label,
            x + (w - lw) * 0.5,
            y + 14.0,
            scale,
            [255, 255, 255, 255],
            &self.render.font,
        );
    }

    /// Handle a click in the block picker overlay.
    pub(crate) fn handle_block_picker_click(&mut self) {
        let reg = self.world_state.world.registry();
        let block_count = reg.count();
        let cols = 9;
        let slot = 40.0;
        let gap = 2.0;
        let (w, h) = self.render.window_size;
        let grid_w = cols as f32 * (slot + gap) - gap;
        let rows = block_count.div_ceil(cols);
        let grid_h = rows as f32 * (slot + gap) - gap;
        let x0 = (w as f32 - grid_w) * 0.5;
        let y0 = (h as f32 - grid_h) * 0.5;

        let mx = self.gameplay.mouse_pos.0;
        let my = self.gameplay.mouse_pos.1;

        // Check if click is within the grid.
        if mx >= x0 && mx < x0 + grid_w && my >= y0 && my < y0 + grid_h {
            let col = ((mx - x0) / (slot + gap)) as usize;
            let row = ((my - y0) / (slot + gap)) as usize;
            let idx = row * cols + col;
            if idx < block_count {
                let id = voxel_core::BlockId(idx as u16);
                self.gameplay.hotbar.set_slot(self.gameplay.hotbar.selected, id);
                let name = reg.get(id).name.as_ref();
                self.gameplay.chat.push_message(format!("Selected: {name}"));
                self.gameplay.block_picker_open = false;
                self.lock_cursor();
            }
        }
    }

    /// Handle a click in the pause menu.
    pub(crate) fn handle_pause_click(&mut self) {
        if let Some(buttons) = self.gameplay.pause_buttons {
            // Back to Game button.
            if self.point_in_rect(
                self.gameplay.mouse_pos,
                buttons[0].0,
                buttons[0].1,
                buttons[0].2,
                buttons[0].3,
            ) {
                self.enter_playing();
            }
            // Exit Game button.
            if self.point_in_rect(
                self.gameplay.mouse_pos,
                buttons[1].0,
                buttons[1].1,
                buttons[1].2,
                buttons[1].3,
            ) {
                log::info!("exit game requested");
                self.gameplay.want_exit = true;
            }
        }
    }

    /// Draw the block picker overlay: a grid of all available blocks.
    fn draw_block_picker(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let reg = self.world_state.world.registry();
        let block_count = reg.count();
        let cols = 9;
        let slot = 40.0;
        let gap = 2.0;
        let rows = block_count.div_ceil(cols);
        let grid_w = cols as f32 * (slot + gap) - gap;
        let grid_h = rows as f32 * (slot + gap) - gap;
        let x0 = (w - grid_w) * 0.5;
        let y0 = (h - grid_h) * 0.5;

        // Background overlay.
        ui.quad(0.0, 0.0, w, h, [0, 0, 0, 160]);
        // Panel background.
        ui.quad(
            x0 - 8.0,
            y0 - 24.0,
            grid_w + 16.0,
            grid_h + 32.0,
            [30, 30, 30, 230],
        );

        // Title.
        ui.text(
            "Block Picker (E to close)",
            x0,
            y0 - 18.0,
            1.0,
            [200, 200, 200, 255],
            &self.render.font,
        );

        for i in 0..block_count {
            let col = i % cols;
            let row = i / cols;
            let x = x0 + col as f32 * (slot + gap);
            let y = y0 + row as f32 * (slot + gap);

            let id = voxel_core::BlockId(i as u16);
            let def = reg.get(id);

            // Slot background.
            ui.quad(x, y, slot, slot, [50, 50, 50, 200]);
            ui.rect_border(x, y, slot, slot, 1.0, [100, 100, 100, 220]);

            // Block name (abbreviated). Both branches align to `&str` because
            // `def.name` derefs `Arc<str>` -> `str` and `&def.name[..6]` is
            // already `&str`.
            let name = if def.name.len() > 6 {
                &def.name[..6]
            } else {
                def.name.as_ref()
            };
            ui.text(
                name,
                x + 2.0,
                y + slot - 10.0,
                0.8,
                [220, 220, 220, 255],
                &self.render.font,
            );
        }
    }

    /// Draw the chat overlay: message history + input line.
    fn draw_chat(&self, ui: &mut UiDrawData, _w: f32, h: f32) {
        let line_h = 16.0;
        let max_visible = 10;
        let pad = 8.0;

        let visible_messages = self.gameplay.chat.messages.len().min(max_visible);
        let input_line = if self.gameplay.chat.open { 1 } else { 0 };
        let total_lines = visible_messages + input_line;

        if total_lines == 0 {
            return;
        }

        let box_h = total_lines as f32 * line_h + pad * 2.0;
        let box_w = 400.0;
        let box_x = pad;
        let box_y = h - box_h - 60.0;

        ui.quad(box_x, box_y, box_w, box_h, [0, 0, 0, 160]);

        let mut y = box_y + pad;
        for i in (0..visible_messages).rev() {
            if let Some(msg) = self.gameplay.chat.messages.get(i) {
                ui.text(msg, box_x + pad, y, 1.0, [200, 200, 200, 255], &self.render.font);
                y += line_h;
            }
        }

        if self.gameplay.chat.open {
            let input_text = format!("> {}", self.gameplay.chat.input_buf);
            ui.text(
                &input_text,
                box_x + pad,
                y,
                1.0,
                [255, 255, 100, 255],
                &self.render.font,
            );
        }
    }

    /// Draw the F3 debug overlay.
    fn draw_debug_overlay(&self, ui: &mut UiDrawData, _w: f32, _h: f32) {
        let x = 8.0;
        let mut y = 8.0;
        let line_h = 16.0;
        let color = [255, 255, 255, 230];

        let panel_w = 320.0;
        let panel_h = 9.0 * line_h + 16.0;
        ui.quad(x - 4.0, y - 4.0, panel_w, panel_h, [0, 0, 0, 150]);

        let pos = GamePlayState::player_pos(&self.ecs_world).unwrap_or(self.gameplay.player.pos);
        let flying = GamePlayState::player_flying(&self.ecs_world);
        let lines = [
            format!("XYZ: {:.1} / {:.1} / {:.1}", pos.x, pos.y, pos.z),
            format!(
                "Chunk: {} / {} / {}",
                (pos.x as i32) >> 4,
                (pos.y as i32) >> 4,
                (pos.z as i32) >> 4
            ),
            format!(
                "Chunks GPU: {}",
                self.render.renderer.as_ref().map(|r| r.chunk_count()).unwrap_or(0)
            ),
            format!("Loaded: {}", self.world_state.world.loaded_chunk_count()),
            format!("Meshed: {}", self.world_state.world.meshed_chunk_count()),
            format!(
                "Time: {:.1}s / {:.0}s",
                self.gameplay.game_time, self.gameplay.day_length
            ),
            format!("Fly: {}", if flying { "ON" } else { "OFF" }),
            format!(
                "Wireframe: {}",
                self.render
                    .renderer
                    .as_ref()
                    .map(|r| r.is_wireframe())
                    .unwrap_or(false)
            ),
        ];

        for line in &lines {
            ui.text(line, x, y, 1.0, color, &self.render.font);
            y += line_h;
        }

        // Chunk debug mini-map (F7).
        if self.gameplay.chunk_debug_enabled {
            self.draw_chunk_debug_minimap(ui, _w, _h);
        }
    }

    /// Draw a mini-map of nearby chunk states.
    fn draw_chunk_debug_minimap(&self, ui: &mut UiDrawData, w: f32, h: f32) {
        let map_x = w - 200.0;
        let map_y = h - 220.0;
        let map_w = 192.0;
        let map_h = 192.0;

        // Background.
        ui.quad(
            map_x - 4.0,
            map_y - 4.0,
            map_w + 8.0,
            map_h + 8.0,
            [0, 0, 0, 180],
        );
        ui.text(
            "Chunk Debug",
            map_x,
            map_y - 20.0,
            1.0,
            [200, 200, 255, 230],
            &self.render.font,
        );

        let pos = GamePlayState::player_pos(&self.ecs_world).unwrap_or(self.gameplay.player.pos);
        let player_chunk_x = (pos.x as i32) >> 4;
        let player_chunk_z = (pos.z as i32) >> 4;
        let half = 6;
        let cell = map_w / (half * 2 + 1) as f32;

        let center = voxel_core::math::ChunkPos::new(player_chunk_x, 0, player_chunk_z);
        let batch = self.world_state.world.chunk_debug_info_batch(center, half);

        for (pos, loaded, dirty, palette_mode, has_mesh) in &batch {
            let color = if *loaded {
                if *dirty {
                    [255, 100, 100, 200]
                } else if !*has_mesh {
                    [255, 255, 100, 200]
                } else if *palette_mode {
                    [100, 100, 255, 200]
                } else {
                    [100, 255, 100, 200]
                }
            } else {
                [60, 60, 60, 150]
            };

            let dx = pos.x() - player_chunk_x;
            let dz = pos.z() - player_chunk_z;
            let sx = map_x + (dx + half) as f32 * cell;
            let sy = map_y + (dz + half) as f32 * cell;
            ui.quad(sx, sy, cell - 1.0, cell - 1.0, color);
        }

        // Player indicator.
        let px = map_x + half as f32 * cell + cell * 0.25;
        let py = map_y + half as f32 * cell + cell * 0.25;
        ui.quad(px, py, cell * 0.5, cell * 0.5, [255, 255, 255, 255]);
    }

    /// Draw the profiler overlay (F6) on the right side of the screen.
    fn draw_profiler_overlay(&self, ui: &mut UiDrawData, w: f32, _h: f32) {
        let line_h = 16.0;
        let pad = 8.0;
        let panel_w = 340.0;
        let panel_h = 16.0 * line_h + pad * 2.0;
        let panel_x = w - panel_w - pad;
        let panel_y = pad;

        ui.quad(panel_x, panel_y, panel_w, panel_h, [0, 0, 0, 160]);

        let white = [255, 255, 255, 230];
        let green = [100, 255, 100, 230];
        let yellow = [255, 255, 100, 230];
        let red = [255, 100, 100, 230];

        let x = panel_x + pad;
        let mut y = panel_y + pad;

        // CPU stats.
        let avg_cpu = self.profiler.avg_ms();
        let fps = self.profiler.avg_fps();
        let fps_color = if fps >= 55.0 {
            green
        } else if fps >= 30.0 {
            yellow
        } else {
            red
        };
        ui.text(
            &format!("FPS: {:.0}  ({:.1} ms)", fps, avg_cpu),
            x,
            y,
            1.0,
            fps_color,
            &self.render.font,
        );
        y += line_h;

        // GPU stats.
        if let Some(latest) = self.profiler.gpu_timings.back() {
            let gpu_color = if latest.frame_ms < 16.0 {
                green
            } else if latest.frame_ms < 33.0 {
                yellow
            } else {
                red
            };
            ui.text(
                &format!("GPU: {:.2} ms", latest.frame_ms),
                x,
                y,
                1.0,
                gpu_color,
                &self.render.font,
            );
            y += line_h;

            let total = latest.frame_ms.max(0.001);
            ui.text(
                &format!(
                    "  Sky:       {:.2} ms ({:.0}%)",
                    latest.sky_ms,
                    latest.sky_ms / total * 100.0
                ),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
            ui.text(
                &format!(
                    "  Opaque:    {:.2} ms ({:.0}%)",
                    latest.opaque_ms,
                    latest.opaque_ms / total * 100.0
                ),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
            ui.text(
                &format!(
                    "  Trans.:    {:.2} ms ({:.0}%)",
                    latest.transparent_ms,
                    latest.transparent_ms / total * 100.0
                ),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
            ui.text(
                &format!(
                    "  UI:        {:.2} ms ({:.0}%)",
                    latest.ui_ms,
                    latest.ui_ms / total * 100.0
                ),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
            ui.text(
                &format!(
                    "  Shadow:    {:.2} ms ({:.0}%)",
                    latest.shadow_ms,
                    latest.shadow_ms / total * 100.0
                ),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
            ui.text(
                &format!(
                    "  Post:      {:.2} ms ({:.0}%)",
                    latest.post_ms,
                    latest.post_ms / total * 100.0
                ),
                x,
                y,
                1.0,
                white,
                &self.render.font,
            );
            y += line_h;
        } else {
            ui.text(
                "GPU: waiting...",
                x,
                y,
                1.0,
                [150, 150, 150, 200],
                &self.render.font,
            );
            y += line_h;
            y += line_h * 6.0;
        }

        // Average GPU over history.
        if !self.profiler.gpu_timings.is_empty() {
            let n = self.profiler.gpu_timings.len();
            let avg_gpu: f32 =
                self.profiler.gpu_timings.iter().map(|t| t.frame_ms).sum::<f32>() / n as f32;
            let max_gpu: f32 = self
                .profiler
                .gpu_timings
                .iter()
                .map(|t| t.frame_ms)
                .fold(0.0f32, f32::max);
            ui.text(
                &format!("Avg GPU: {:.2} ms  Max: {:.2} ms", avg_gpu, max_gpu),
                x,
                y,
                1.0,
                [180, 180, 255, 230],
                &self.render.font,
            );
            y += line_h;
        }

        // Bar chart of last 60 frames.
        let chart_x = x;
        let chart_y = y + 4.0;
        let chart_w = panel_w - pad * 2.0;
        let chart_h = 40.0;
        ui.quad(chart_x, chart_y, chart_w, chart_h, [30, 30, 30, 200]);
        let bar_count = self.profiler.gpu_timings.len().min(60);
        if bar_count > 0 {
            let bar_w = chart_w / 60.0;
            let start = self.profiler.gpu_timings.len().saturating_sub(60);
            let max_ms = self
                .profiler
                .gpu_timings
                .iter()
                .skip(start)
                .map(|t| t.frame_ms)
                .fold(1.0f32, f32::max);
            for (i, timing) in self.profiler.gpu_timings.iter().skip(start).enumerate() {
                let bar_h = (timing.frame_ms / max_ms * chart_h).max(1.0);
                let bx = chart_x + i as f32 * bar_w;
                let by = chart_y + chart_h - bar_h;
                let c = if timing.frame_ms < 16.0 {
                    [80, 200, 80, 220]
                } else if timing.frame_ms < 33.0 {
                    [220, 220, 80, 220]
                } else {
                    [220, 80, 80, 220]
                };
                ui.quad(bx, by, bar_w - 1.0, bar_h, c);
            }
        }
    }
}
