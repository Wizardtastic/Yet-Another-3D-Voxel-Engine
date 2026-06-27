// Compiles the GLSL chunk shaders to SPIR-V at build time using glslangValidator
// from the Vulkan SDK. The resulting .spv files land in OUT_DIR and are
// include_bytes!'d by the renderer, so no compiled shaders are committed.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // The shaders live at the workspace root under `shaders/`.
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let shader_dir = PathBuf::from(&manifest)
        .join("..")
        .join("..")
        .join("shaders");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let glslang = find_glslang_validator();
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("chunk.vert").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("chunk.frag").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("ui.vert").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("ui.frag").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("sky.vert").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("sky.frag").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("shadow.vert").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("shadow.frag").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("post.vert").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        shader_dir.join("post.frag").display()
    );
    println!("cargo:rerun-if-changed=build.rs");

    let shaders = [
        ("chunk.vert", "chunk.vert.spv"),
        ("chunk.frag", "chunk.frag.spv"),
        ("ui.vert", "ui.vert.spv"),
        ("ui.frag", "ui.frag.spv"),
        ("sky.vert", "sky.vert.spv"),
        ("sky.frag", "sky.frag.spv"),
        ("shadow.vert", "shadow.vert.spv"),
        ("shadow.frag", "shadow.frag.spv"),
        ("post.vert", "post.vert.spv"),
        ("post.frag", "post.frag.spv"),
    ];

    for (src, dst) in shaders {
        let src_path = shader_dir.join(src);
        let dst_path = out_dir.join(dst);
        if !src_path.exists() {
            panic!("shader source not found: {}", src_path.display());
        }
        let status = Command::new(&glslang)
            .arg("-V") // compile to SPIR-V (Vulkan target)
            .arg("-o")
            .arg(&dst_path)
            .arg(&src_path)
            .status()
            .unwrap_or_else(|e| panic!("failed to run glslangValidator: {e}"));
        if !status.success() {
            panic!("glslangValidator failed to compile {}", src_path.display());
        }
        println!("cargo:rerun-if-changed={}", src_path.display());
    }
}

fn find_glslang_validator() -> PathBuf {
    if let Ok(sdk) = env::var("VULKAN_SDK") {
        let candidate = Path::new(&sdk).join("Bin").join("glslangValidator.exe");
        if candidate.exists() {
            return candidate;
        }
        let candidate2 = Path::new(&sdk).join("Bin").join("glslangValidator");
        if candidate2.exists() {
            return candidate2;
        }
    }
    // PATH fallback.
    if let Ok(out) = Command::new("where").arg("glslangValidator").output() {
        if out.status.success() {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(line) = s.lines().next() {
                return PathBuf::from(line.trim());
            }
        }
    }
    panic!("glslangValidator not found; set VULKAN_SDK or add it to PATH");
}
