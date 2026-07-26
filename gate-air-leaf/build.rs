//! Build script for the gate_air K1/K4 tracegen CUDA kernels (AOT, ahead-of-time).
//!
//! The two tracegen kernels — K1 `cuda/gate_sim.cu` (prog_slot_meta / gate_sim_states / gate_sim)
//! and K4 `cuda/interaction.cu` (logup_* + ps_* prefix-sum) — are compiled HERE with nvcc into
//! multi-arch fatbins in `$OUT_DIR`, replacing the old runtime NVRTC (`compile_ptx`) path. Each .cu
//! is FULLY SELF-CONTAINED (no `#include`, no external `__device__` symbols — they define their own
//! m31/cm31/qm31 helpers inline), so — unlike the sibling gate-air-cuda-kernel — this needs no
//! `-dc`/`-dlink`/device-linking/static-lib/fields.cu/objcopy. Just `nvcc -fatbin` per .cu; the Rust
//! loads each fatbin via cudarc's `Ptx::from_file` + `load_ptx` (see gpu_tracegen.rs).
//!
//! Flags mirror the sibling build.rs / stwo's CMake: `-std=c++17 --expt-relaxed-constexpr
//! --ptxas-options=-v`, arch auto-detected via nvidia-smi (fallback 89;100;120). `-fatbin` (not
//! `-cubin`) so the multi-arch fallback works and cudarc's `cuModuleLoad` can load it.
//!
//! Only built under `--features gpu-cuda` (needs nvcc). `STWO_CUDA_SKIP_BUILD=1` skips the native
//! compile so the Rust can be `cargo check`ed on a laptop without nvcc (mirrors stwo's escape hatch).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/gate_sim.cu");
    println!("cargo:rerun-if-changed=cuda/interaction.cu");
    println!("cargo:rerun-if-env-changed=STWO_CUDA_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");

    // The tracegen module (gpu_tracegen.rs) is gated on the `gpu-cuda` cargo feature. A default /
    // non-cuda build never references the fatbins, so a plain `cargo check` must NOT invoke nvcc.
    if std::env::var_os("CARGO_FEATURE_GPU_CUDA").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    std::process::exit(0);

    // Allow type-checking a `--features gpu-cuda`/`cuda` build without nvcc (laptop). Paired with
    // stwo's own STWO_CUDA_SKIP_BUILD — set both on the laptop, neither on the GPU box.
    if std::env::var_os("STWO_CUDA_SKIP_BUILD").is_some() {
        println!(
            "cargo:warning=STWO_CUDA_SKIP_BUILD set — skipping gate_air tracegen CUDA native build \
             (check-only)"
        );
        return;
    }

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_src_dir = PathBuf::from(&manifest_dir).join("cuda");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let arches = cuda_architectures();
    let gencode_flags: Vec<String> = arches
        .iter()
        .flat_map(|a| {
            vec![
                "-gencode".to_string(),
                format!("arch=compute_{a},code=sm_{a}"),
            ]
        })
        .collect();

    // Compile each self-contained .cu to a loadable multi-arch fatbin in $OUT_DIR.
    for (src, out) in [
        ("gate_sim.cu", "gate_sim.fatbin"),
        ("interaction.cu", "interaction.fatbin"),
    ] {
        let src_path = cuda_src_dir.join(src);
        let out_path = out_dir.join(out);
        let mut cmd = Command::new("nvcc");
        cmd.args(&gencode_flags)
            .arg("-std=c++17")
            .arg("--expt-relaxed-constexpr")
            .arg("--ptxas-options=-v")
            .arg("-fatbin")
            .arg(&src_path)
            .arg("-o")
            .arg(&out_path);
        run(&mut cmd, "nvcc compile fatbin");
    }
}

/// Determine the CUDA architectures to `-gencode` for, mirroring the CMakeLists auto-detect:
/// honor `CMAKE_CUDA_ARCHITECTURES` if set, else query `nvidia-smi`, else fall back to 89;100;120.
fn cuda_architectures() -> Vec<String> {
    if let Ok(v) = std::env::var("CMAKE_CUDA_ARCHITECTURES") {
        let parsed: Vec<String> = v
            .split(';')
            .flat_map(|s| s.split(','))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    // nvidia-smi --query-gpu=compute_cap --format=csv,noheader -> e.g. "8.9"
    if let Ok(out) = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
    {
        if out.status.success() {
            if let Some(first) = String::from_utf8_lossy(&out.stdout).lines().next() {
                let cap = first.trim().replace('.', "");
                if !cap.is_empty() && cap.chars().all(|c| c.is_ascii_digit()) {
                    return vec![cap];
                }
            }
        }
    }
    vec!["89".to_string(), "100".to_string(), "120".to_string()]
}

fn run(cmd: &mut Command, what: &str) {
    println!("cargo:warning=gate_air tracegen CUDA {what}: {cmd:?}");
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {what} ({cmd:?}): {e}"));
    if !status.success() {
        panic!("{what} failed (exit {status}): {cmd:?}");
    }
}
