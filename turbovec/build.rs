// Emit BLAS link directives so downstream binaries (tests, examples, and
// crate users) resolve `cblas_sgemm` without needing `extern crate blas_src;`.
//
// ndarray's `blas` feature calls into C-BLAS for matrix multiplication but
// doesn't pick a provider; the provider lives in an external native library
// (OpenBLAS on Linux, Apple's Accelerate on macOS). Emitting the link flag
// from this crate's build script attaches the directive to `turbovec` itself,
// so any binary that depends on `turbovec` inherits it — bypassing the
// "blas-src must be referenced in the final binary" footgun.
//
// Windows falls through to ndarray's pure-Rust matrixmultiply fallback.
//
// === Static BLAS for npm consumers ===
// Python wheels bundle libopenblas.so via auditwheel; npm has no equivalent
// tool, so a dynamically-linked .node addon would fail to load on an end-user
// machine that does not have system OpenBLAS installed. To keep the .node
// self-contained we support env-gated static linking on Linux:
//
//   TURBOVEC_STATIC_BLAS=1   — link OpenBLAS statically instead of dynamically
//   OPENBLAS_LIB_DIR=<path>  — search this directory for libopenblas.a
//
// Both flags are intentionally OFF by default so that the existing Python
// wheel builds, `cargo install`, and downstream crate users are completely
// unaffected. Only the Node.js CI/release job sets them.
fn main() {
    // Tell Cargo to re-run this script if the env vars change.
    println!("cargo:rerun-if-env-changed=TURBOVEC_STATIC_BLAS");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");

    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => {
            // When TURBOVEC_STATIC_BLAS is set (to any non-empty value), link
            // OpenBLAS statically so the resulting .node is self-contained.
            // If OPENBLAS_LIB_DIR is also set, add it to the native search path
            // (useful in CI where libopenblas.a lives in a non-standard prefix).
            // When TURBOVEC_STATIC_BLAS is unset the behaviour is byte-identical
            // to the original build script: dynamic linking, no search-path hint.
            if std::env::var("TURBOVEC_STATIC_BLAS")
                .map(|v| !v.is_empty())
                .unwrap_or(false)
            {
                if let Ok(lib_dir) = std::env::var("OPENBLAS_LIB_DIR") {
                    println!("cargo:rustc-link-search=native={lib_dir}");
                }
                println!("cargo:rustc-link-lib=static=openblas");
            } else {
                println!("cargo:rustc-link-lib=openblas");
            }
        }
        Ok("macos") => println!("cargo:rustc-link-lib=framework=Accelerate"),
        _ => {}
    }
}
