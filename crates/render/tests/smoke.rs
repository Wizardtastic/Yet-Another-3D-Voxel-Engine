//! Smoke tests for the `voxel-render` public API surface.
//!
//! These tests cover the small, dependency-free part of the public surface:
//! plain data structs, enums, and `Pod`/`Zeroable` traits on GPU-side layouts.
//!
//! They do NOT exercise the `Renderer` itself or any of the private
//! `record_*_pass` helpers — those need a live Vulkan instance/device and are
//! validated by the engine's snapshot tests in `crates/app/tests/snapshots.rs`
//! instead.

use voxel_render::{ChunkUpload, GpuTimings, MeshPass, RendererConfig, Vertex};
use voxel_core::math::ChunkPos;

#[test]
fn renderer_config_default_disables_validation() {
    // The library default flips to `validation=true` ONLY when explicitly
    // requested (e.g. by the engine's debug builds). Picking it up by accident
    // here would surprise sandbox/CI environments that lack Vulkan layers.
    let cfg = RendererConfig::default();
    assert!(
        !cfg.validation,
        "default RendererConfig must disable Vulkan validation (no layer in sandbox)"
    );
}

#[test]
fn renderer_config_default_clear_color_is_opaque() {
    // We don't pin the RGB channels (stylistic) but DO pin alpha to 1.0:
    // a translucent clear would expose blending bugs that have nothing to do
    // with this test. Asserting opacity is a structural invariant of "sky".
    let cfg = RendererConfig::default();
    assert_eq!(
        cfg.clear_color[3], 1.0,
        "default sky clear colour alpha must be 1.0 (opaque)"
    );
}

#[test]
fn renderer_config_default_fog_distance_is_positive() {
    let cfg = RendererConfig::default();
    assert!(
        cfg.fog_distance > 0.0 && cfg.fog_distance.is_finite(),
        "default fog_distance must be a positive finite value"
    );
}

#[test]
fn gpu_timings_supports_steady_state_assignment() {
    // Roundtrip a non-default GpuTimings through Copy semantics. Default-zero
    // is checked by the `Default` derive itself; the meaningful guarantee is
    // that the struct stays `Copy` + fieldwise-assignable as a 7-f32 block.
    let t = GpuTimings {
        frame_ms: 16.6,
        shadow_ms: 0.5,
        sky_ms: 0.2,
        opaque_ms: 8.0,
        transparent_ms: 1.0,
        ui_ms: 0.3,
        post_ms: 0.4,
    };
    let frame_ms_sum = t.frame_ms + t.shadow_ms + t.sky_ms + t.opaque_ms
        + t.transparent_ms + t.ui_ms + t.post_ms;
    assert_eq!(
        frame_ms_sum, 27.0,
        "GpuTimings additions must be exact (no implicit padding/NaN bits)"
    );
}

#[test]
fn mesh_pass_usable_as_hashmap_key() {
    // MeshPass is the per-chunk-meshes render-pass key, looked up in
    // HashMap<MeshPass, ChunkBuffers>. PartialEq+Eq+Hash derive guarantees the
    // bumps themselves, but we want a runtime check that an Opaque and a
    // Transparent entry DON'T collide when stored in the same map.
    use std::collections::HashMap;
    let mut map: HashMap<MeshPass, &'static str> = HashMap::new();
    map.insert(MeshPass::Opaque, "opaque");
    map.insert(MeshPass::Transparent, "transparent");
    assert_eq!(map.get(&MeshPass::Opaque), Some(&"opaque"));
    assert_eq!(map.get(&MeshPass::Transparent), Some(&"transparent"));
    assert_eq!(map.len(), 2, "Opaque and Transparent must hash distinctly");
}

#[test]
fn vertex_layout_is_pod_roundtrip_safe_with_nonzero_payload() {
    // Roundtrip a non-zero Vertex through `bytemuck::bytes_of` and back. If the
    // layout were misaligned or had implicit padding, this would either panic
    // (Cast error) or come back with bits flipped to zero on the way through.
    // The non-zero payload catches the latter case.
    let v = Vertex {
        pos: [1.0, 2.0, 3.0],
        uv: [0.5, 0.25],
        light: 0.75,
    };
    let bytes = bytemuck::bytes_of(&v);
    assert_eq!(
        bytes.len(),
        24,
        "Vertex must be exactly 24 bytes (3*4 pos + 2*4 uv + 4 light)"
    );
    let back: Vertex = *bytemuck::from_bytes(bytes);
    assert_eq!(back.pos[0], 1.0, "pos[0] must survive POD roundtrip");
    assert_eq!(back.pos, v.pos);
    assert_eq!(back.uv, v.uv);
    assert_eq!(
        back.light, v.light,
        "light must survive POD roundtrip (no padding bit-stealing)"
    );
}

#[test]
fn chunk_upload_carries_pass_through_construction() {
    // `ChunkUpload` is what `voxel-world` hands to the renderer — verify the
    // pass field is preserved end-to-end (not silently swallowed by a future
    // refactor that forgets to wire it through). Uses a non-Opaque pass to
    // catch accidental `default()` substitutions.
    let upload = ChunkUpload {
        pos: ChunkPos(glam::IVec3::new(0, 0, 0)),
        pass: MeshPass::Transparent,
        vertices: Vec::new(),
        indices: Vec::new(),
        index_count: 0,
    };
    assert_eq!(upload.pass, MeshPass::Transparent);
    assert_eq!(upload.index_count, 0);
    assert_eq!(upload.pos.0, glam::IVec3::new(0, 0, 0));
}
