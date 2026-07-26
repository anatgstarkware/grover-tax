//! Build script for the gate_air device-resident CUDA constraint kernel.
//!
//! This crate is the DOWNSTREAM plugin that owns the gate_air-specific CUDA kernel. The generic
//! `stwo-cuda-backend` (libstwo_cuda) is now gate_air-free; the two gate_air translation units
//! (`cuda/evaluate_gate_air.cu` + `cuda/gate_air_entry.cu`) are compiled HERE into their own static
//! library `libgate_air_cuda.a`, which HOST-LINKS against libstwo_cuda.
//!
//! # Device-link scope: gate_air's TUs + fields.cu (host-link for the finalize kernel)
//! Two classes of cross-library symbol, resolved differently:
//!
//! 1. HOST-launched kernel — `generic_constraint_quotients_finalize_kernel` (libstwo_cuda's
//!    `evaluate_common.cu`). gate_air HOST-launches it (`<<<...>>>`), so it's a plain external
//!    `__global__` launch-stub that resolves at HOST-LINK against libstwo_cuda.a — like the sibling
//!    TUs that already `U`-reference it. No device-link needed for this.
//!
//! 2. `__device__` FIELD ARITHMETIC — `add`/`sub`/`mul`/`neg` on m31/qm31 (declared in `fields.cuh`,
//!    DEFINED in libstwo_cuda's `fields.cu`, NOT header-inline). gate_air's DEVICE code calls these,
//!    so nvlink must resolve them at device-link time. We therefore compile `fields.cu` (from the
//!    shared source dir) with `-dc`, feed its object to the `-dlink` step (embedding the field device
//!    code into the self-contained `gate_air_dlink.o` fatbin), AND archive `fields.o` into
//!    libgate_air_cuda.a — because the `-dlink` registration ctor references fields.o's host-side
//!    fatbin wrapper. fields.o also carries the `__host__` copies of the field arithmetic, but since
//!    libstwo_cuda.a is linked plain `static` (NOT +whole-archive) and both fields.o objects export
//!    the identical strong symbol set, on-demand archive resolution pulls only one — no
//!    multiple-definition. This keeps gate_air a self-contained DEVICE module.
//!
//! `fields.cu` is fully self-contained (all its helpers — inv/div/pow/high_as_m31/… — are defined
//! within it), so it introduces no further device deps. The remaining headers (`utils.cuh`,
//! `logup.cuh`, `eval_at_row.cuh`, `evaluate_common.cuh`, `timer.cuh`) are header-inline for the
//! device symbols gate_air actually calls (nvlink reported ONLY the field ops as unresolved).
//!   * NO `-rdc`/device-link against libstwo_cuda.a,
//!   * NO CUDA_RESOLVE_DEVICE_SYMBOLS change / relocatable-archive rebuild on libstwo_cuda,
//!   * NO header-inlining of fields.
//!
//! # Flags mirror stwo-cuda-backend's CMakeLists.txt EXACTLY
//! See `stwo-cuda-backend/crates/stwo/src/stwo_cuda/cuda/CMakeLists.txt`:
//!   * CUDA/C++ standard 17               (lines 2-3)
//!   * `--expt-relaxed-constexpr`         (line 35)
//!   * Release flags empty (no -O relic) (line 36)
//!   * `-Xcompiler -fPIC`                 (line 239: `target_compile_options(... -fPIC)`)
//!   * `--ptxas-options=-v`               (line 241)
//!   * arch: auto-detect via nvidia-smi, fallback 89;100;120 (lines 7-32)
//! ABI/arch compatibility with libstwo_cuda depends on matching arch + std; keep these in sync if
//! the CMake flags change.
//!
//! Only built under `--features cuda` (needs nvcc). `STWO_CUDA_SKIP_BUILD=1` skips the native
//! compile+link so the Rust can be `cargo check`ed on a laptop without nvcc (mirrors stwo's
//! build.rs escape hatch).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Match stwo's build.rs: the CUDA kernel is only built under `--features cuda`.
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    #[cfg(target_os = "macos")]
    std::process::exit(0);

    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_src_dir = PathBuf::from(&manifest_dir).join("cuda");

    // Re-run if any of our CUDA sources change.
    println!("cargo:rerun-if-changed=build.rs");
    println!(
        "cargo:rerun-if-changed={}",
        cuda_src_dir.join("evaluate_gate_air.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_src_dir.join("evaluate_gate_air.cuh").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        cuda_src_dir.join("gate_air_entry.cu").display()
    );
    println!("cargo:rerun-if-env-changed=STWO_CUDA_SKIP_BUILD");
    println!("cargo:rerun-if-env-changed=CMAKE_CUDA_ARCHITECTURES");

    // Allow type-checking without nvcc (laptop `cargo check`). Paired with stwo's own
    // STWO_CUDA_SKIP_BUILD — set both on the laptop, neither on the GPU box.
    if std::env::var_os("STWO_CUDA_SKIP_BUILD").is_some() {
        println!(
            "cargo:warning=STWO_CUDA_SKIP_BUILD set — skipping gate_air CUDA native build \
             (check-only)"
        );
        return;
    }

    // The shared CUDA header root (fields.cuh, evaluate_common.cuh, ...) exported by the patched
    // `stwo` crate (links = "stwo_cuda"): its build.rs emits `cargo:include=<.../cuda>` which cargo
    // surfaces to this DIRECT dependent as DEP_STWO_CUDA_INCLUDE.
    let stwo_cuda_include = std::env::var("DEP_STWO_CUDA_INCLUDE").expect(
        "DEP_STWO_CUDA_INCLUDE not set — the `stwo` dependency (patched to stwo-cuda-backend) must \
         expose its CUDA include dir via `links = \"stwo_cuda\"`; check the [patch] in \
         gate-air-leaf/Cargo.toml and that stwo is built with --features cuda.",
    );
    // fields.cu is borrowed from the shared source dir for device-link; rebuild if it changes.
    println!(
        "cargo:rerun-if-changed={}",
        Path::new(&stwo_cuda_include).join("fields.cu").display()
    );

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

    // Compile a TU with `-dc` (relocatable device code) into `out_dir`. `src_dir` differs for
    // gate_air's own TUs (cuda_src_dir) vs the borrowed fields.cu (stwo_cuda_include).
    let compile_dc = |src_dir: &Path, src: &str| -> PathBuf {
        let src_path = src_dir.join(src);
        let obj_path = out_dir.join(format!("{src}.o"));
        let mut cmd = Command::new("nvcc");
        cmd.arg("-std=c++17")
            .arg("--expt-relaxed-constexpr")
            .arg("-dc") // relocatable device code (needed for device-link below)
            .args(&gencode_flags)
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("--ptxas-options=-v")
            // gate_air's own dir (sibling evaluate_gate_air.cuh) + the shared stwo CUDA headers.
            .arg("-I")
            .arg(&cuda_src_dir)
            .arg("-I")
            .arg(&stwo_cuda_include)
            .arg("-c")
            .arg(&src_path)
            .arg("-o")
            .arg(&obj_path);
        run(&mut cmd, "nvcc compile");
        obj_path
    };

    // gate_air's own two TUs — these ARE archived (host code links against libstwo_cuda).
    let objs = ["evaluate_gate_air.cu", "gate_air_entry.cu"]
        .iter()
        .map(|src| compile_dc(&cuda_src_dir, src))
        .collect::<Vec<_>>();

    // fields.cu — borrowed from the shared source dir purely to satisfy gate_air's DEVICE-side
    // field-arithmetic refs at device-link. Compiled with -dc, fed ONLY to `-dlink` below, and
    // deliberately NOT archived (its __host__ copies would collide with libstwo_cuda.a).
    let fields_obj = compile_dc(Path::new(&stwo_cuda_include), "fields.cu");

    // Device-link gate_air's objects + fields.cu into one relocatable object. This resolves
    // gate_air's inter-TU device refs AND its device-side field-arithmetic refs (add/sub/mul/neg on
    // m31/qm31), embedding the resulting device image self-contained in gate_air_dlink.o. The
    // HOST-launched finalize kernel is NOT device-linked here — it resolves at the final host link
    // against libstwo_cuda.a.
    let dlink_obj = out_dir.join("gate_air_dlink.o");
    {
        let mut cmd = Command::new("nvcc");
        cmd.args(&gencode_flags)
            .arg("-Xcompiler")
            .arg("-fPIC")
            .arg("-dlink")
            .args(&objs)
            .arg(&fields_obj)
            .arg("-o")
            .arg(&dlink_obj);
        run(&mut cmd, "nvcc device-link");
    }

    // fields.o MUST be archived so the `-dlink` registration ctor in gate_air_dlink.o can resolve
    // fields.o's host-side fatbin wrapper (`__fatbinwrap_..._fields_cu`). BUT fields.o also carries
    // __host__ copies of every field-arithmetic function (mul/add/sub/neg/inv/… on m31/cm31/qm31),
    // and libstwo_cuda.a — which its own device-registration force-includes — defines the SAME strong
    // host symbols → multiple-definition at the final host link. So we localize fields.o's field
    // symbols, keeping ONLY the CUDA registration/fatbin symbols global (the ones gate_air_dlink.o
    // needs). gate_air's own host-side field refs then resolve to libstwo_cuda's global copies.
    // Localize ONLY the C++-mangled field functions (`_Z…` — mul/add/sub/neg/inv/… on m31/cm31/qm31,
    // the exact symbols that collide with libstwo_cuda.a). Everything else — the CUDA runtime/fatbin
    // registration machinery (`__cuda*`, `__nv*`, `__fatbin*`) that gate_air_dlink.o's ctor needs —
    // stays global. (A blanket `--keep-global-symbol` allowlist risked localizing a needed CUDA
    // internal symbol, breaking device-module registration → segfault at first kernel launch.)
    let fields_local = out_dir.join("fields_local.o");
    {
        let mut cmd = Command::new("objcopy");
        cmd.arg("-w")
            .arg("--localize-symbol=_Z*")
            .arg(&fields_obj)
            .arg(&fields_local);
        run(&mut cmd, "objcopy localize fields host symbols");
    }

    // Archive gate_air's TUs + the localized fields.o + the device-link object into libgate_air_cuda.a.
    let lib_path = out_dir.join("libgate_air_cuda.a");
    let _ = std::fs::remove_file(&lib_path);
    {
        let mut cmd = Command::new("ar");
        cmd.arg("crs").arg(&lib_path);
        for o in &objs {
            cmd.arg(o);
        }
        cmd.arg(&fields_local);
        cmd.arg(&dlink_obj);
        run(&mut cmd, "ar archive");
    }

    // Link this crate against the gate_air kernel archive. libstwo_cuda + cudart + stdc++ are
    // already linked by the `stwo` crate's build.rs (DEP_STWO_CUDA_LIB_DIR); the external
    // `generic_constraint_quotients_finalize_kernel` resolves there at host-link. rustc uses the
    // last-writer-wins order, but static archives resolve by symbol so ordering is handled by the
    // final link's --start-group semantics under cc. We additionally re-emit cudart/stdc++ (and the
    // CUDA device-runtime for the -dlink object) to be safe.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gate_air_cuda");
    // The -dlink object needs the CUDA device runtime; harmless if already linked by stwo.
    let cuda_lib_path = Path::new("/usr/local/cuda/lib64");
    if cuda_lib_path.exists() {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_lib_path.display()
        );
    }
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=cudart");
    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-lib=stdc++");
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
    // Fallback matches CMakeLists lines 29-30.
    vec!["89".to_string(), "100".to_string(), "120".to_string()]
}

fn run(cmd: &mut Command, what: &str) {
    println!("cargo:warning=gate_air CUDA {what}: {cmd:?}");
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {what} ({cmd:?}): {e}"));
    if !status.success() {
        panic!("{what} failed (exit {status}): {cmd:?}");
    }
}
