//! `voxel-engine` — application shell.
//!
//! Owns the winit window, the Vulkan [`Renderer`], the shared [`World`], the
//! background [`ChunkStreamer`], and the gameplay state ([`Player`], [`Hotbar`],
//! [`InputState`]). Drives a fixed-timestep simulation with interpolated
//! rendering, translates raw window events into resolved input, and streams
//! chunk meshes to the GPU as the world loads.
//!
//! This is also the future host for the plugin/scripting runtime and the
//! dedicated-server entry point (same shell, headless renderer).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, ElementState, Ime, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

use crate::keybinds::physical_key_to_char;

use voxel_game::input::Action;
use voxel_game::{
    input_system, lifecycle_system, movement_system, Aabb,
    CameraOwner,    CameraResource, ChatState, Hotbar,
    InputResource,    InputSnapshot, InputState, Player, PlayerConfig,
    PlayerEntity, PlayerInput, PlayerState, Transform, Velocity,
};
use voxel_ecs::{FnSystem, SystemSchedule, World as EcsWorld};
use voxel_render::{FontAtlas, GpuTimings, Renderer, RendererConfig};
use voxel_world::{ChunkStreamer, StreamConfig, World};

pub mod settings;
mod commands;
mod frame;
mod keybinds;
mod save;
mod ui;

/// Whether the game is playing or showing the pause/exit menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GameState {
    Playing,
    PauseMenu,
}

/// Computed day/night parameters for the current game time.
struct DayParams {
    horizon: [f32; 3],
    zenith: [f32; 3],
    fog: [f32; 3],
    daylight: f32,
    sun_altitude: f32,
    sun_angle: f32,
}

/// Rolling profiler state for the debug overlay.
struct ProfilerState {
    frame_times_ms: VecDeque<f64>,
    gpu_timings: VecDeque<GpuTimings>,
    enabled: bool,
    /// Frame start time captured by `begin_frame`.
    frame_start: Option<Instant>,
    /// Last minimap-update timestamp (reserved for future rate-limiting).
    last_minimap_update: Option<Instant>,
}

impl Default for ProfilerState {
    fn default() -> Self {
        Self {
            frame_times_ms: VecDeque::with_capacity(120),
            gpu_timings: VecDeque::with_capacity(120),
            enabled: false,
            frame_start: None,
            last_minimap_update: None,
        }
    }
}

impl ProfilerState {
    /// Record the start of a new frame.
    fn begin_frame(&mut self) {
        self.frame_start = Some(Instant::now());
    }

    /// Record frame completion and push the frame time (in ms) into the
    /// rolling 120-frame history.
    fn end_frame(&mut self, frame_time_ms: f64) {
        self.frame_times_ms.push_back(frame_time_ms);
        if self.frame_times_ms.len() > 120 {
            self.frame_times_ms.pop_front();
        }
    }

    /// Average frame time over the rolling window (in ms).
    fn avg_ms(&self) -> f64 {
        if self.frame_times_ms.is_empty() {
            0.0
        } else {
            self.frame_times_ms.iter().sum::<f64>() / self.frame_times_ms.len() as f64
        }
    }

    /// Average FPS over the rolling window.
    fn avg_fps(&self) -> f64 {
        let avg = self.avg_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }
}

/// Render-related state: the winit window, the Vulkan renderer, window size,
/// and the bitmap font used for UI text.
struct RenderState {
    pub renderer: Option<Renderer>,
    pub window: Option<Window>,
    pub window_size: (u32, u32),
    pub font: FontAtlas,
}

impl RenderState {
    fn new(_config: &EngineConfig) -> Self {
        Self {
            renderer: None,
            window: None,
            window_size: (1280, 720),
            font: FontAtlas::new(),
        }
    }

    fn resize(&mut self) {
        if let Some(r) = self.renderer.as_mut() {
            r.resize();
        }
    }
}

/// Engine-level input & timing state. Embeds the gameplay [`InputState`] and
/// holds cursor lock, keybind map, per-frame timing, and scratch buffers.
struct EngineInputState {
    /// Game-level input state (held actions, clicks, mouse delta).
    pub input: InputState,
    /// True when the OS cursor is locked for FPS look.
    pub cursor_locked: bool,
    /// Last frame timestamp used for delta computation.
    pub last_time: Instant,
    /// Last frame delta in seconds.
    pub frame_time: f64,
    /// Fixed-timestep simulation accumulator.
    pub sim_accumulator: f64,
    /// Total frames rendered (used for auto-capture).
    pub frame_count: usize,
    /// True once the auto-capture screenshot has been written.
    pub captured: bool,
    /// True when a screenshot has been requested and should be taken in frame().
    pub screenshot_requested: bool,
    /// True after the window has been created and the render loop started.
    pub running: bool,
    /// True once the player's spawn chunk has loaded; physics is paused until then.
    pub spawned: bool,
    /// Resolved keybind map: KeyCode → Action.
    pub keybinds: settings::KeybindMap,
    /// Last sun_dir sent to streamer (avoid redundant sends).
    pub last_sun_dir: Vec3,
    /// Last player position sent to streamer focus (avoid redundant sends).
    pub last_focus_pos: Vec3,
}

impl EngineInputState {
    fn new(keybinds: settings::KeybindMap) -> Self {
        Self {
            input: InputState::default(),
            cursor_locked: false,
            last_time: Instant::now(),
            frame_time: 0.016,
            sim_accumulator: 0.0,
            frame_count: 0,
            captured: false,
            screenshot_requested: false,
            running: false,
            spawned: false,
            keybinds,
            last_sun_dir: Vec3::ZERO,
            last_focus_pos: Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        }
    }
}

/// World state: the shared [`World`] handle and the background [`ChunkStreamer`].
struct WorldState {
    pub world: Arc<World>,
    pub streamer: Option<ChunkStreamer>,
}

impl WorldState {
    fn new(world: Arc<World>) -> Self {
        Self {
            world,
            streamer: None,
        }
    }
}

/// Gameplay state: player, hotbar, chat, undo/redo, pause/menu flags, etc.
pub(crate) struct GamePlayState {
    /// Player entity (position, velocity, camera, etc.).
    pub player: Player,
    /// 9-slot hotbar.
    pub hotbar: Hotbar,
    /// Current game state (playing vs pause menu).
    pub game_state: GameState,
    /// Game time in seconds (wraps at day_length).
    pub game_time: f64,
    /// Length of a full day/night cycle in seconds.
    pub day_length: f64,
    /// Mouse position in physical pixels (for pause-menu and block-picker hit-testing).
    pub mouse_pos: (f32, f32),
    /// Pre-computed pause-menu button rects: (back_btn, exit_btn) in pixels.
    pub pause_buttons: Option<[(f32, f32, f32, f32); 2]>,
    /// Chat and command system.
    pub chat: ChatState,
    /// Undo/redo stack for block edits.
    pub undo_redo: voxel_game::UndoRedoState,
    /// Block picker (inventory) open.
    pub block_picker_open: bool,
    /// Schematic clipboard: ((x1,y1,z1), (x2,y2,z2), blocks)
    pub clipboard: Option<Clipboard>,
    /// Debug overlay enabled (F3 toggle).
    pub debug_overlay: bool,
    /// Chunk debug visualization enabled (F7 toggle).
    pub chunk_debug_enabled: bool,
    /// Set by the Exit Game button; checked in about_to_wait to exit the loop.
    pub want_exit: bool,
}

impl GamePlayState {
    fn new(player: Player, hotbar: Hotbar, day_length: f64) -> Self {
        Self {
            player,
            hotbar,
            game_state: GameState::Playing,
            game_time: 300.0, // start at dawn
            day_length,
            mouse_pos: (0.0, 0.0),
            pause_buttons: None,
            chat: ChatState::default(),
            undo_redo: voxel_game::UndoRedoState::default(),
            block_picker_open: false,
            clipboard: None,
            debug_overlay: false,
            chunk_debug_enabled: false,
            want_exit: false,
        }
    }

    // -----------------------------------------------------------------
    // ECS-backed player accessors.
    //
    // The legacy `Player` struct on `GamePlayState` is kept as a stub for
    // backwards compatibility, but the *source of truth* is now the entity
    // in the ECS world. These helpers read from the ECS and return
    // `None` if the player has not been spawned yet.
    // -----------------------------------------------------------------

    /// Resolve the player entity handle from the ECS resource.
    pub fn player_entity(ecs: &EcsWorld) -> Option<voxel_ecs::Entity> {
        ecs.resource::<PlayerEntity>().and_then(|p| p.0)
    }

    /// Read the player's world-space position from the ECS, if available.
    pub fn player_pos(ecs: &EcsWorld) -> Option<Vec3> {
        let e = Self::player_entity(ecs)?;
        ecs.get::<Transform>(e).map(|t| t.pos)
    }

    /// Read the player's linear velocity from the ECS, if available.
    pub fn player_vel(ecs: &EcsWorld) -> Option<Vec3> {
        let e = Self::player_entity(ecs)?;
        ecs.get::<Velocity>(e).map(|v| v.lin)
    }

    /// Read whether the player is currently flying (ECS state).
    pub fn player_flying(ecs: &EcsWorld) -> bool {
        Self::player_entity(ecs)
            .and_then(|e| ecs.get::<PlayerInput>(e))
            .map(|i| i.flying)
            .unwrap_or(false)
    }

    /// Set the player's flying flag in the ECS.
    pub fn set_player_flying(ecs: &mut EcsWorld, flying: bool) {
        if let Some(e) = Self::player_entity(ecs) {
            if let Some(input) = ecs.get_mut::<PlayerInput>(e) {
                input.flying = flying;
            }
        }
    }

    /// Set the player's world-space position in the ECS. Also clears
    /// vertical velocity so the player doesn't immediately fall after
    /// a teleport.
    pub fn set_player_pos(ecs: &mut EcsWorld, pos: Vec3) {
        if let Some(e) = Self::player_entity(ecs) {
            if let Some(t) = ecs.get_mut::<Transform>(e) {
                t.pos = pos;
            }
            if let Some(v) = ecs.get_mut::<Velocity>(e) {
                v.lin = Vec3::ZERO;
            }
        }
    }

    /// Compute the current eye position (head height) by reading the
    /// player's `PlayerState` eye offset from the ECS.
    pub fn player_eye_pos(ecs: &EcsWorld) -> Option<Vec3> {
        let e = Self::player_entity(ecs)?;
        let t = ecs.get::<Transform>(e)?;
        let s = ecs.get::<PlayerState>(e).copied().unwrap_or_default();
        Some(t.pos + glam::Vec3::new(0.0, s.eye_offset, 0.0))
    }

    /// Read the player state component from the ECS, if available.
    #[allow(dead_code)]
    pub fn player_state(ecs: &EcsWorld) -> Option<PlayerState> {
        let e = Self::player_entity(ecs)?;
        ecs.get::<PlayerState>(e).copied()
    }

    /// Construct a fresh `voxel_core::Camera` from the ECS state. This
    /// is the same view used by the renderer.
    pub fn player_camera(ecs: &EcsWorld) -> Option<voxel_core::Camera> {
        let cam_res = ecs.resource::<CameraResource>()?;
        Some(cam_res.0)
    }
}

/// Engine configuration.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    /// World generation seed.
    pub seed: i32,
    /// Window title.
    pub title: String,
    /// Initial window size in physical pixels.
    pub window_size: (u32, u32),
    /// Render configuration.
    pub render: RendererConfig,
    /// Chunk streaming configuration.
    pub stream: StreamConfig,
    /// Player configuration.
    pub player: PlayerConfig,
    /// If set, automatically capture a screenshot after this many frames and
    /// save it to `capture_path` (used for headless verification).
    pub capture_after_frames: Option<usize>,
    /// Where to write an auto-capture screenshot.
    pub capture_path: PathBuf,
    /// Exit the process shortly after the auto-capture completes.
    pub exit_after_capture: bool,
    /// Spawn position (world space). If `None`, a surface height is found.
    pub spawn: Option<[f32; 3]>,
    /// Day length in seconds.
    pub day_length: f64,
    /// Keybind settings from config.
    pub keybinds: settings::KeybindSettings,
    /// Path to the assets directory (for data-driven block definitions).
    pub assets_path: Option<PathBuf>,
    /// Enable cascaded shadow mapping.
    pub shadow_enabled: bool,
    /// Shadow map resolution per cascade (texels per side).
    pub shadow_resolution: u32,
    /// HDR tonemapping exposure.
    pub exposure: f32,
    /// Vignette darkening strength at screen edges.
    pub vignette_strength: f32,
    /// Open the window in borderless-fullscreen mode (instead of windowed).
    pub fullscreen: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            seed: 1337,
            title: "voxel — custom Vulkan engine".into(),
            window_size: (1280, 720),
            render: RendererConfig::default(),
            stream: StreamConfig::default(),
            player: PlayerConfig::default(),
            capture_after_frames: None,
            capture_path: PathBuf::from("capture.png"),
            exit_after_capture: false,
            spawn: None,
            day_length: 1200.0,
            keybinds: settings::KeybindSettings::default(),
            assets_path: None,
            shadow_enabled: true,
            shadow_resolution: 2048,
            exposure: 1.0,
            vignette_strength: 0.3,
            fullscreen: false,
        }
    }
}

/// Entry point: create the event loop and run the engine until the window
/// closes (or until an auto-capture + exit completes).
pub fn run(config: EngineConfig) -> Result<()> {
    let event_loop = EventLoop::new().map_err(|e| anyhow!("EventLoop::new: {e}"))?;
    let mut app = EngineApp::new(config)?;
    event_loop
        .run_app(&mut app)
        .map_err(|e| anyhow!("event loop ended with error: {e}"))?;
    Ok(())
}


/// Schematic clipboard: origin corner, opposite corner, flattened block list.
type Clipboard = ((i32, i32, i32), (i32, i32, i32), Vec<voxel_core::BlockId>);

pub(crate) struct EngineApp {
    config: EngineConfig,
    /// Render state: window, renderer, font, window size.
    render: RenderState,
    /// Engine input & timing: cursor lock, keybinds, frame timing, scratch buffers.
    input: EngineInputState,
    /// World state: shared world handle and chunk streamer.
    world_state: WorldState,
    /// ECS world owning entities (player), resources (camera, input, player
    /// handle), and the system schedule that runs each fixed step.
    ecs_world: EcsWorld,
    /// Compiled gameplay system schedule. `None` until `EngineApp::new`
    /// finishes initialising.
    schedule: Option<SystemSchedule>,
    /// Gameplay state: player, hotbar, chat, game state, undo/redo, etc.
    gameplay: GamePlayState,
    /// Profiler state (rolling frame time, GPU timings).
    profiler: ProfilerState,
}

impl EngineApp {
    fn new(config: EngineConfig) -> Result<Self> {
        let world = World::new_with_path(config.seed, config.assets_path.as_deref());
        // Find a land spawn (above sea level) using the height function directly
        // — no chunk loading needed. Falls back to a high spawn at origin.
        let (sx, sy, sz) = world.terrain().find_spawn();
        let spawn = config
            .spawn
            .unwrap_or([sx as f32 + 0.5, sy as f32 + 2.0, sz as f32 + 0.5]);
        log::info!(
            "spawn search: land column at ({}, {}, {}) -> spawn pos {:?}",
            sx,
            sy,
            sz,
            spawn
        );
        let day_length = config.day_length;
        let player = Player::new(Vec3::from(spawn), config.player);
        let mut hotbar = Hotbar::new();
        hotbar.populate_defaults(&world.registry());
        let keybinds = config.keybinds.resolve();
        let render = RenderState::new(&config);

        // --- ECS world + resources ---
        let mut ecs_world = EcsWorld::new();
        let spawn_pos = Vec3::from(spawn);
        // Insert a default camera resource — `update_camera_from_transform` will
        // keep this in sync with the player transform each frame.
        let mut initial_camera = voxel_core::Camera::default();
        initial_camera.pos = spawn_pos + glam::Vec3::new(0.0, voxel_game::EYE_HEIGHT, 0.0);
        ecs_world.insert_resource(CameraResource(initial_camera));
        ecs_world.insert_resource(InputResource(InputSnapshot::default()));
        // Spawn the player entity with the full set of components gameplay
        // systems expect.
        let player_entity = ecs_world.spawn((
            Transform {
                pos: spawn_pos,
                rot: glam::Quat::IDENTITY,
            },
            Velocity::default(),
            Aabb::default(),
            PlayerInput::default(),
            PlayerState::default(),
            CameraOwner,
        ));
        ecs_world.insert_resource(PlayerEntity(Some(player_entity)));

        // --- Build the gameplay system schedule ---
        let schedule = SystemSchedule::new()
            .add_system(FnSystem::new("InputSystem", input_system))
            .add_system(FnSystem::new("MovementSystem", movement_system))
            .add_system(FnSystem::new("LifecycleSystem", lifecycle_system));

        Ok(Self {
            config,
            render,
            input: EngineInputState::new(keybinds),
            world_state: WorldState::new(world),
            ecs_world,
            schedule: Some(schedule),
            gameplay: GamePlayState::new(player, hotbar, day_length),
            profiler: ProfilerState::default(),
        })
    }

    fn lock_cursor(&mut self) {
        if let Some(w) = &self.render.window {
            let _ = w
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| w.set_cursor_grab(CursorGrabMode::Confined));
            w.set_cursor_visible(false);
            self.input.cursor_locked = true;
        }
    }

    fn unlock_cursor(&mut self) {
        if let Some(w) = &self.render.window {
            let _ = w.set_cursor_grab(CursorGrabMode::None);
            w.set_cursor_visible(true);
            self.input.cursor_locked = false;
        }
    }

    /// Compute day/night parameters from game_time.
    fn day_params(&self) -> DayParams {
        let t = (self.gameplay.game_time % self.gameplay.day_length) / self.gameplay.day_length; // 0..1
        let sun_angle = (t as f32) * std::f32::consts::TAU;
        let sun_altitude = (sun_angle - std::f32::consts::FRAC_PI_2).sin(); // -1..1
        let daylight = (sun_altitude * 1.2 + 0.6).clamp(0.0, 1.0);

        // Day sky colours.
        let day_horizon = [0.62, 0.78, 0.95];
        let day_zenith = [0.35, 0.55, 0.90];
        let day_fog = [0.62, 0.80, 0.96];
        // Night sky colours.
        let night_horizon = [0.05, 0.06, 0.12];
        let night_zenith = [0.01, 0.02, 0.05];
        let night_fog = [0.04, 0.05, 0.10];

        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };

        DayParams {
            horizon: lerp(night_horizon, day_horizon, daylight),
            zenith: lerp(night_zenith, day_zenith, daylight),
            fog: lerp(night_fog, day_fog, daylight),
            daylight,
            sun_altitude,
            sun_angle,
        }
    }

    /// Transition to Playing state (lock cursor, resume sim).
    fn enter_playing(&mut self) {
        self.gameplay.game_state = GameState::Playing;
        self.lock_cursor();
    }

    /// Transition to PauseMenu state (unlock cursor, pause sim).
    fn enter_pause(&mut self) {
        self.gameplay.game_state = GameState::PauseMenu;
        self.unlock_cursor();
        self.input.input.held.clear();
        self.gameplay.block_picker_open = false;
        self.gameplay.chat.open = false;
        self.input.sim_accumulator = 0.0;    }
}

impl ApplicationHandler for EngineApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.render.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title(self.config.title.clone())
            .with_inner_size(PhysicalSize::new(
                self.config.window_size.0,
                self.config.window_size.1,
            ))
            .with_fullscreen(if self.config.fullscreen {
                Some(Fullscreen::Borderless(None))
            } else {
                None
            });
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                log::error!("create_window: {e}");
                self.gameplay.want_exit = true;
                return;
            }
        };
        self.render.window = Some(window);

        // Build the renderer from the window's raw handles.
        if let Some(window) = &self.render.window {
            use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
            let wh = match window.window_handle() {
                Ok(h) => h.as_raw(),
                Err(e) => {
                    log::error!("window_handle: {e}");
                    self.gameplay.want_exit = true;
                    return;
                }
            };
            let dh = match window.display_handle() {
                Ok(h) => h.as_raw(),
                Err(e) => {
                    log::error!("display_handle: {e}");
                    self.gameplay.want_exit = true;
                    return;
                }
            };
            match Renderer::new(wh, dh, self.config.render.clone()) {
                Ok(r) => {
                    self.render.window_size = (r.extent().width, r.extent().height);
                    self.render.renderer = Some(r);
                }
                Err(e) => {
                    log::error!("Renderer::new: {e}");
                    self.gameplay.want_exit = true;
                    return;
                }
            }
        }

        // Spawn the chunk streamer and populate the hotbar.
        let focus_pos =
            GamePlayState::player_pos(&self.ecs_world).unwrap_or(self.gameplay.player.pos);
        match ChunkStreamer::spawn(self.world_state.world.clone(), self.config.stream) {
            Ok(s) => {
                s.set_focus(focus_pos);
                self.world_state.streamer = Some(s);
            }
            Err(e) => {
                log::error!("ChunkStreamer::spawn: {e}");
                self.gameplay.want_exit = true;
                return;
            }
        }

        // Lock the cursor for first-person look.
        self.lock_cursor();
        self.input.running = true;
        self.input.last_time = Instant::now();

        // Request the first redraw to kick-start the render loop.
        if let Some(w) = &self.render.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::Resized(size) => {
                self.render.window_size = (size.width, size.height);
                self.render.resize();
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::PhysicalKey;
                let pressed = event.state == ElementState::Pressed;

                // Chat input takes priority when open.
                if self.gameplay.chat.open {
                    if pressed {
                        if let PhysicalKey::Code(code) = event.physical_key {
                            match code {
                                KeyCode::Escape => {
                                    self.gameplay.chat.close();
                                    self.lock_cursor();
                                    if let Some(w) = &self.render.window {
                                        w.set_ime_allowed(false);
                                    }
                                }
                                KeyCode::Enter => {
                                    let pos = GamePlayState::player_pos(&self.ecs_world)
                                        .unwrap_or(self.gameplay.player.pos);
                                    let result = self.gameplay.chat.submit_with_pos(pos);
                                    self.execute_command(result);
                                    self.lock_cursor();
                                    if let Some(w) = &self.render.window {
                                        w.set_ime_allowed(false);
                                    }
                                }
                                KeyCode::Backspace => {
                                    self.gameplay.chat.backspace();
                                }
                                KeyCode::Space => {
                                    self.gameplay.chat.push_char(' ');
                                }
                                KeyCode::Tab => {
                                    self.gameplay.chat.tab_complete();
                                }
                                KeyCode::ArrowUp => {
                                    self.gameplay.chat.history_up();
                                }
                                KeyCode::ArrowDown => {
                                    self.gameplay.chat.history_down();
                                }
                                _ => {
                                    // Map physical key codes to characters directly.
                                    if let Some(ch) = physical_key_to_char(code) {
                                        self.gameplay.chat.push_char(ch);
                                    }
                                }
                            }
                        }
                    }
                    return;
                }

                if let PhysicalKey::Code(code) = event.physical_key {
                    // Ctrl+Z / Ctrl+Y for undo/redo (check before other handlers).
                    if pressed && self.input.input.held(Action::Sprint) {
                        match code {
                            KeyCode::KeyZ => {
                                if let Some(action) = self.gameplay.undo_redo.pop_undo() {
                                    for edit in action.edits.iter().rev() {
                                        let id = voxel_core::BlockId(edit.old_block);
                                        self.world_state.world.set_block(edit.x, edit.y, edit.z, id);
                                    }
                                    self.gameplay.chat.push_message(format!(
                                        "Undid {} block changes",
                                        action.edits.len()
                                    ));
                                }
                                return;
                            }
                            KeyCode::KeyY => {
                                if let Some(action) = self.gameplay.undo_redo.pop_redo() {
                                    for edit in &action.edits {
                                        let id = voxel_core::BlockId(edit.new_block);
                                        self.world_state.world.set_block(edit.x, edit.y, edit.z, id);
                                    }
                                    self.gameplay.chat.push_message(format!(
                                        "Redid {} block changes",
                                        action.edits.len()
                                    ));
                                }
                                return;
                            }
                            _ => {}
                        }
                    }

                    // Movement actions (held): directly mapped.
                    let action = match code {
                        KeyCode::KeyW => Some(Action::Forward),
                        KeyCode::KeyS => Some(Action::Back),
                        KeyCode::KeyA => Some(Action::Left),
                        KeyCode::KeyD => Some(Action::Right),
                        KeyCode::Space => Some(Action::Jump),
                        KeyCode::ShiftLeft => Some(Action::Sneak),
                        KeyCode::ControlLeft => Some(Action::Sprint),
                        _ => None,
                    };

                    // Non-movement keybinds: looked up from config.
                    if let Some(&bound_action) = self.input.keybinds.get(&code) {
                        if pressed {
                            match bound_action {
                                Action::DebugOverlay => {
                                    self.gameplay.debug_overlay = !self.gameplay.debug_overlay;
                                }
                                Action::Wireframe => {
                                    if let Some(r) = self.render.renderer.as_mut() {
                                        r.toggle_wireframe();
                                    }
                                }
                                Action::Fly => {
                                    let new_flying = !GamePlayState::player_flying(&self.ecs_world);
                                    // Legacy stub mirror.
                                    self.gameplay.player.flying = new_flying;
                                    // ECS source of truth.
                                    GamePlayState::set_player_flying(&mut self.ecs_world, new_flying);
                                    log::info!(
                                        "fly mode: {}",
                                        if new_flying { "ON" } else { "OFF" }
                                    );
                                }
                                Action::Chat => {
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.gameplay.chat.open();
                                        self.unlock_cursor();
                                        self.input.input.held.clear();
                                        if let Some(w) = &self.render.window {
                                            w.set_ime_allowed(true);
                                        }
                                    }
                                }
                                Action::Screenshot => {
                                    self.input.screenshot_requested = true;
                                }
                                Action::Pause => {
                                    self.gameplay.block_picker_open = false;
                                    self.gameplay.chat.open = false;
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.enter_pause();
                                    } else {
                                        self.enter_playing();
                                    }
                                }
                                Action::RenderDistanceUp => {
                                    let r = (self.config.stream.load_radius + 1).min(16);
                                    self.config.stream.load_radius = r;
                                    if let Some(s) = &self.world_state.streamer {
                                        s.set_load_radius(r as u32);
                                    }
                                    self.gameplay.chat.push_message(format!("Render distance: {r}"));
                                }
                                Action::RenderDistanceDown => {
                                    let r = (self.config.stream.load_radius - 1).max(2);
                                    self.config.stream.load_radius = r;
                                    if let Some(s) = &self.world_state.streamer {
                                        s.set_load_radius(r as u32);
                                    }
                                    self.gameplay.chat.push_message(format!("Render distance: {r}"));
                                }
                                Action::BlockPicker => {
                                    if self.gameplay.game_state == GameState::Playing {
                                        self.gameplay.block_picker_open = !self.gameplay.block_picker_open;
                                        if self.gameplay.block_picker_open {
                                            self.unlock_cursor();
                                            self.input.input.held.clear();
                                        } else {
                                            self.lock_cursor();
                                        }
                                    }
                                }
                                Action::Profiler => {
                                    self.profiler.enabled = !self.profiler.enabled;
                                }
                                Action::ChunkDebug => {
                                    self.gameplay.chunk_debug_enabled = !self.gameplay.chunk_debug_enabled;
                                    self.gameplay.chat.push_message(format!(
                                        "Chunk debug: {}",
                                        if self.gameplay.chunk_debug_enabled {
                                            "ON"
                                        } else {
                                            "OFF"
                                        }
                                    ));
                                }
                                _ => {}
                            }
                        }
                    } else if pressed {
                        // Escape is handled separately (not in keybind map) to
                        // avoid accidental re-binding.
                        if code == KeyCode::Escape {
                            if self.gameplay.block_picker_open {
                                self.gameplay.block_picker_open = false;
                                self.lock_cursor();
                            } else if self.gameplay.game_state == GameState::Playing {
                                self.enter_pause();
                            } else {
                                self.enter_playing();
                            }
                        }
                        // Hotbar slot selection (Digit1-9) — always hardcoded.
                        let slot = match code {
                            KeyCode::Digit1 => Some(0),
                            KeyCode::Digit2 => Some(1),
                            KeyCode::Digit3 => Some(2),
                            KeyCode::Digit4 => Some(3),
                            KeyCode::Digit5 => Some(4),
                            KeyCode::Digit6 => Some(5),
                            KeyCode::Digit7 => Some(6),
                            KeyCode::Digit8 => Some(7),
                            KeyCode::Digit9 => Some(8),
                            _ => None,
                        };
                        if let Some(idx) = slot {
                            self.gameplay.hotbar.select(idx);
                        }
                    }

                    if let Some(a) = action {
                        if pressed {
                            self.input.input.held.insert(a);
                        } else {
                            self.input.input.held.remove(&a);
                        }
                    }
                }
            }
            WindowEvent::Ime(Ime::Commit(text)) => {
                if self.gameplay.chat.open {
                    for ch in text.chars() {
                        self.gameplay.chat.push_char(ch);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                if self.gameplay.game_state == GameState::Playing {
                    if !self.input.cursor_locked && pressed {
                        // Clicking the window re-locks the cursor for mouse look.
                        // This fixes "losing mouse movement when I click on it".
                        self.lock_cursor();
                        return;
                    }
                    if self.input.cursor_locked {
                        match button {
                            MouseButton::Left => self.input.input.clicks.left = pressed,
                            MouseButton::Right => self.input.input.clicks.right = pressed,
                            _ => {}
                        }
                    }
                } else if self.gameplay.game_state == GameState::PauseMenu {
                    // In pause menu, clicks are handled by handle_pause_click in frame().
                    if pressed {
                        match button {
                            MouseButton::Left => self.input.input.clicks.left = true,
                            MouseButton::Right => self.input.input.clicks.right = true,
                            _ => {}
                        }
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                // Track mouse position for pause-menu button hit-testing.
                self.gameplay.mouse_pos = (position.x as f32, position.y as f32);
            }
            WindowEvent::RedrawRequested => {
                if self.input.running {
                    self.frame();
                }
            }
            WindowEvent::Focused(false)
                // Auto-pause when the window loses focus — but not in capture
                // mode (headless verification), where the window may not have focus.
                if self.gameplay.game_state == GameState::Playing
                    && self.config.capture_after_frames.is_none() =>
            {
                self.enter_pause();
            }
            _ => {}
        }

        // Auto-exit after capture, if configured.
        if self.config.exit_after_capture && self.input.captured {
            event_loop.exit();
        }
        // Exit game button.
        if self.gameplay.want_exit {
            event_loop.exit();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if !self.input.cursor_locked {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.input.mouse_delta.0 += delta.0 as f32;
            self.input.input.mouse_delta.1 += delta.1 as f32;
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.render.window {
            if self.input.running && !self.gameplay.want_exit {
                w.request_redraw();
            }
        }
    }
}
