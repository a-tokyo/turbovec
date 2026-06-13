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
                // Also add the GCC runtime library search paths so the linker
                // can find libgfortran.a and libgomp.a.  We ask GCC to print
                // its own library directories and add each one that exists.
                // If the helper isn't available we silently skip (non-Linux
                // cross-compile environments where TURBOVEC_STATIC_BLAS is
                // never set anyway).
                if let Ok(output) = std::process::Command::new("gcc")
                    .args(["-print-search-dirs"])
                    .output()
                {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    for line in stdout.lines() {
                        if let Some(rest) = line.strip_prefix("libraries: =") {
                            for dir in rest.split(':') {
                                let trimmed = dir.trim();
                                if !trimmed.is_empty()
                                    && std::path::Path::new(trimmed).exists()
                                {
                                    println!("cargo:rustc-link-search=native={trimmed}");
                                }
                            }
                        }
                    }
                }

                // Static OpenBLAS — must come FIRST so it can pull in symbols
                // from the Fortran/OpenMP runtimes listed after it.
                println!("cargo:rustc-link-lib=static=openblas");

                // Fortran runtime: ship libgfortran statically so the .node
                // doesn't require a system gfortran install on end-user boxes.
                // (libgfortran.a is present in the Debian/Ubuntu gfortran package.)
                println!("cargo:rustc-link-lib=static=gfortran");

                // OpenMP runtime used by OpenBLAS's threaded kernels: link
                // statically for the same self-contained-binary reason.
                println!("cargo:rustc-link-lib=static=gomp");

                // quadmath (128-bit float support) is x86_64-only; on aarch64
                // it doesn't exist.  Attempt static link; if the .a isn't
                // present the linker simply won't need it (OpenBLAS aarch64
                // doesn't reference __float128 symbols).
                // NOTE: on x86_64 hosts add `cargo:rustc-link-lib=static=quadmath`
                // if the link step fails with undefined `__quadmath_*` symbols.

                // pthreads and libm are present on every Linux box — keep them
                // dynamic so we don't pull in a second copy from musl/glibc.
                println!("cargo:rustc-link-lib=pthread");
                println!("cargo:rustc-link-lib=m");
            } else {
                println!("cargo:rustc-link-lib=openblas");
            }
        }
        Ok("macos") => println!("cargo:rustc-link-lib=framework=Accelerate"),
        _ => {}
    }
}
