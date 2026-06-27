//! Snapshot / golden-image tests for the voxel engine.
//!
//! These tests spawn the engine binary, render a fixed number of frames with a
//! deterministic seed, and compare the captured screenshot against a golden
//! reference image stored in `tests/snapshots/`.
//!
//! All tests in this module are `#[ignore]` because they require a real GPU
//! and display. Run with:
//!
//! ```sh
//! cargo test -p voxel-app -- --ignored
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;

/// Number of frames to render before capturing.
const CAPTURE_FRAMES: usize = 60;

/// Maximum per-pixel color distance (0–255) before a pixel is considered
/// different. Accounts for minor floating-point / driver variations.
const PIXEL_TOLERANCE: u8 = 4;

/// Maximum fraction of pixels that may differ before the test fails.
const DIFF_THRESHOLD: f64 = 0.01; // 1 %

fn snapshot_dir() -> PathBuf {
    // The test lives in crates/app/tests/snapshots.rs;
    // golden images live in crates/app/tests/snapshots/.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/snapshots")
}

fn binary_path() -> PathBuf {
    // cargo places test binaries next to the real binary; find `voxel.exe` via
    // the OUT_DIR / CARGO_BIN_EXE environment variable.  Fallback: assume
    // the workspace target/debug directory.
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_voxel") {
        return PathBuf::from(path);
    }
    // Fallback for `cargo test` which doesn't set that var.
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Go up from crates/app → workspace root.
    p.pop();
    p.pop();
    p.push("target");
    p.push("debug");
    p.push(if cfg!(windows) { "voxel.exe" } else { "voxel" });
    p
}

/// Run the engine binary with `--capture <frames>` and `--seed <seed>`.
/// Returns the path to the saved screenshot, or an error.
fn capture_screenshot(seed: i32, frames: usize, output: &Path) -> anyhow::Result<()> {
    let bin = binary_path();
    assert!(
        bin.exists(),
        "binary not found at {bin:?} — run `cargo build -p voxel-app` first"
    );

    let status = Command::new(&bin)
        .args([
            "--capture",
            &frames.to_string(),
            "--seed",
            &seed.to_string(),
        ])
        .env("RUST_LOG", "info")
        .current_dir(std::env::current_dir()?)
        .status()?;

    if !status.success() {
        anyhow::bail!("engine exited with {status}");
    }
    if !output.exists() {
        anyhow::bail!("expected screenshot at {output:?} but it was not created");
    }
    Ok(())
}

/// Compare two RGBA images pixel-by-pixel.
/// Returns (total_pixels, different_pixels).
fn compare_images(a: &[u8], b: &[u8]) -> (usize, usize) {
    assert_eq!(a.len(), b.len(), "image sizes differ");
    let mut diff = 0usize;
    // Compare in chunks of 4 (RGBA).
    for chunk in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        let (ra, rb) = chunk;
        for i in 0..4 {
            let diff_val = ra[i].abs_diff(rb[i]);
            if diff_val > PIXEL_TOLERANCE {
                diff += 1;
                break; // count the pixel once even if multiple channels differ
            }
        }
    }
    let total = a.len() / 4;
    (total, diff)
}

/// Load a PNG file as raw RGBA bytes and dimensions.
fn load_rgba(path: &Path) -> anyhow::Result<(u32, u32, Vec<u8>)> {
    let img = image::open(path)?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();
    Ok((w, h, rgba.into_raw()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Smoke test: the engine captures a frame without crashing.
///
/// This does NOT compare against a golden image — it only verifies the capture
/// pipeline works end-to-end on the current machine.
#[test]
#[ignore = "requires GPU and display"]
fn capture_smoke_test() {
    let dir = snapshot_dir();
    std::fs::create_dir_all(&dir).unwrap();
    let out = dir.join("smoke_test.png");

    capture_screenshot(42, CAPTURE_FRAMES, &out).expect("capture failed");

    let (w, h, _rgba) = load_rgba(&out).expect("failed to load captured image");
    assert!(w > 0 && h > 0, "captured image has zero dimensions");
}

/// Golden-image comparison for the default spawn view (seed=1337).
///
/// On first run this creates the golden reference.  On subsequent runs it
/// compares the captured frame against the stored reference.
///
/// To regenerate the golden image, delete `tests/snapshots/golden_1337.png`
/// and re-run this test.
#[test]
#[ignore = "requires GPU and display"]
fn golden_image_seed_1337() {
    let dir = snapshot_dir();
    std::fs::create_dir_all(&dir).unwrap();

    let golden = dir.join("golden_1337.png");
    let out = dir.join("test_output_1337.png");

    capture_screenshot(1337, CAPTURE_FRAMES, &out).expect("capture failed");

    if !golden.exists() {
        // First run — create the golden reference.
        std::fs::copy(&out, &golden).expect("failed to create golden image");
        eprintln!(
            "Created golden reference image at {:?}. Re-run to compare.",
            golden
        );
        return;
    }

    let (gw, gh, g_rgba) = load_rgba(&golden).expect("failed to load golden image");
    let (tw, th, t_rgba) = load_rgba(&out).expect("failed to load test output");

    assert_eq!(
        (gw, gh),
        (tw, th),
        "image dimensions differ: golden {gw}x{gh}, test {tw}x{th}"
    );

    let (total, diff) = compare_images(&g_rgba, &t_rgba);
    let diff_pct = (diff as f64 / total as f64) * 100.0;

    if diff_pct > DIFF_THRESHOLD * 100.0 {
        // Save a diff image for debugging.
        let diff_img = image::RgbaImage::from_fn(gw, gh, |x, y| {
            let idx = ((y * gw + x) * 4) as usize;
            let da = g_rgba[idx];
            let db = t_rgba[idx];
            let dr = da.abs_diff(db);
            let da2 = g_rgba[idx + 1];
            let db2 = t_rgba[idx + 1];
            let dg = da2.abs_diff(db2);
            let da3 = g_rgba[idx + 2];
            let db3 = t_rgba[idx + 2];
            let db3c = da3.abs_diff(db3);
            // Amplify differences for visibility.
            image::Rgba([
                (dr as u16 * 5).min(255) as u8,
                (dg as u16 * 5).min(255) as u8,
                (db3c as u16 * 5).min(255) as u8,
                255,
            ])
        });
        let diff_path = dir.join("diff_1337.png");
        diff_img
            .save(&diff_path)
            .expect("failed to save diff image");

        panic!(
            "Golden image mismatch: {diff}/{total} pixels differ ({diff_pct:.2}%). \
             Diff image saved to {diff_path:?}. \
             If the change is intentional, delete {golden:?} and re-run.",
        );
    }

    // Clean up test output on success.
    let _ = std::fs::remove_file(&out);
}
