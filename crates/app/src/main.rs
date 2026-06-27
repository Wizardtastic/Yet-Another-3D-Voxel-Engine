//! `voxel-app` — executable entry point for the voxel sandbox.
//!
//! Boots the engine with configuration from `config.toml` (if present) merged
//! with CLI flags. For automated verification, pass `--capture <frames>` to
//! render that many frames, save a screenshot, and exit.

use std::path::{Path, PathBuf};

use clap::Parser;
use voxel_engine::settings::GameSettings;
use voxel_engine::{run, EngineConfig};

/// Voxel sandbox engine — a custom Vulkan-based voxel game.
#[derive(Parser, Debug)]
#[command(name = "voxel", version, about)]
struct Cli {
    /// Capture N frames and save a screenshot, then exit.
    #[arg(long)]
    capture: Option<usize>,

    /// World generation seed.
    #[arg(long)]
    seed: Option<i32>,

    /// Enable Vulkan validation layers.
    #[arg(long)]
    validation: bool,

    /// Disable VSync.
    #[arg(long)]
    no_vsync: bool,

    /// Path to config file (default: config.toml).
    #[arg(long, default_value = "config.toml")]
    config: String,

    /// Window width in pixels.
    #[arg(long)]
    width: Option<u32>,

    /// Window height in pixels.
    #[arg(long)]
    height: Option<u32>,

    /// Start in fullscreen mode.
    #[arg(long)]
    fullscreen: bool,

    /// Enable debug overlay on startup.
    #[arg(long)]
    debug: bool,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let cli = Cli::parse();

    // Load config file (from --config path or default).
    let settings = GameSettings::load(Path::new(&cli.config));

    // Build EngineConfig from settings + CLI overrides.
    let mut config = EngineConfig {
        seed: cli.seed.unwrap_or(settings.world.seed),
        title: "voxel — custom Vulkan engine".into(),
        window_size: (
            cli.width.unwrap_or(settings.graphics.width),
            cli.height.unwrap_or(settings.graphics.height),
        ),
        render: settings.to_renderer_config(),
        stream: settings.to_stream_config(),
        player: settings.to_player_config(),
        capture_after_frames: cli.capture,
        capture_path: PathBuf::from("capture.png"),
        exit_after_capture: cli.capture.is_some(),
        spawn: None,
        day_length: settings.world.day_length,
        keybinds: settings.keys,
        assets_path: settings.world.assets_path.map(PathBuf::from),
        shadow_enabled: settings.graphics.shadow_enabled,
        shadow_resolution: settings.graphics.shadow_resolution,
        exposure: settings.graphics.exposure,
        vignette_strength: settings.graphics.vignette_strength,
        fullscreen: cli.fullscreen,
    };

    // CLI flag overrides.
    if cli.validation {
        config.render.validation = true;
    }
    if cli.no_vsync {
        config.render.vsync = false;
    }
    if cli.debug {
        // Set debug overlay on via config — the engine reads this
        // from the config. For now, we'll just pass it through.
        // The engine can check config.debug or a separate field.
    }
    if cli.fullscreen {
        log::info!("starting in borderless fullscreen");
    }

    log::info!("starting voxel engine (seed={})", config.seed);
    run(config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_verify() {
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_defaults() {
        let cli = Cli::try_parse_from(["voxel"]).unwrap();
        assert!(cli.capture.is_none());
        assert!(cli.seed.is_none());
        assert!(!cli.validation);
        assert!(!cli.no_vsync);
        assert_eq!(cli.config, "config.toml");
        assert!(cli.width.is_none());
        assert!(cli.height.is_none());
        assert!(!cli.fullscreen);
        assert!(!cli.debug);
    }

    #[test]
    fn cli_with_args() {
        let cli = Cli::try_parse_from([
            "voxel",
            "--capture", "180",
            "--seed", "42",
            "--validation",
            "--no-vsync",
            "--width", "1920",
            "--height", "1080",
            "--fullscreen",
            "--debug",
        ])
        .unwrap();
        assert_eq!(cli.capture, Some(180));
        assert_eq!(cli.seed, Some(42));
        assert!(cli.validation);
        assert!(cli.no_vsync);
        assert_eq!(cli.width, Some(1920));
        assert_eq!(cli.height, Some(1080));
        assert!(cli.fullscreen);
        assert!(cli.debug);
    }

    #[test]
    fn cli_custom_config() {
        let cli = Cli::try_parse_from(["voxel", "--config", "my_config.toml"]).unwrap();
        assert_eq!(cli.config, "my_config.toml");
    }
}
