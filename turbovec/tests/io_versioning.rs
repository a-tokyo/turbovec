//! Format-versioning tests for `.tv` and `.tvim`.
//!
//! Verifies:
//! 1. Round-trip via the public write/load functions works on the current
//!    format (version 2) for both file types.
//! 2. A hand-constructed version-1 `.tv` file (bare bit_width-first
//!    header, no magic) is rejected with the upgrade-hint error.
//! 3. A hand-constructed version-1 `.tvim` file (TVIM magic with
//!    version byte 1) is rejected with the upgrade-hint error.

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use turbovec::io::{load, load_id_map, write, write_id_map};

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("turbovec-{}-{}", nonce, name));
    p
}

#[test]
fn tv_round_trip_current_format() {
    let path = temp_path("v2.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 3;
    let packed = vec![0xABu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.5f32, 2.5, 3.5];

    // Round-trip with empty TQ+ calibration (identity); behaviour identical
    // to a v2 file otherwise. Separate test below covers populated calibration.
    write(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[]).unwrap();
    let (bw, d, n, p, s, shift, scale_tq) = load(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_round_trip_with_tqplus_calibration() {
    let path = temp_path("v3-tqplus.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 3;
    let packed = vec![0xABu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.5f32, 2.5, 3.5];
    let shift: Vec<f32> = (0..dim).map(|d| d as f32 * 0.01).collect();
    let scale_tq: Vec<f32> = (0..dim).map(|d| 1.0 + d as f32 * 0.02).collect();

    write(&path, bit_width, dim, n_vectors, &packed, &scales, &shift, &scale_tq).unwrap();
    let (bw, d, n, p, s, loaded_shift, loaded_scale) = load(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert_eq!(loaded_shift, shift);
    assert_eq!(loaded_scale, scale_tq);
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_v1_file_is_rejected_with_upgrade_hint() {
    // Hand-construct a turbovec ≤ 0.4.3 `.tv` file: bare header
    // (bit_width=4, dim=32, n_vectors=2), packed codes, two f32 norms.
    let path = temp_path("v1.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(2u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&vec![0u8; (32 / 8) * 4 * 2]).unwrap(); // packed codes
        f.write_all(&(1.0f32).to_le_bytes()).unwrap(); // norm 0
        f.write_all(&(2.0f32).to_le_bytes()).unwrap(); // norm 1
    }

    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("turbovec ≤ 0.4.3") && msg.contains("Rebuild"),
        "expected upgrade hint, got: {}",
        msg
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tvim_round_trip_current_format() {
    let path = temp_path("v2.tvim");
    let bit_width = 2;
    let dim = 16;
    let n_vectors = 4;
    let packed = vec![0x55u8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![0.5f32, 1.0, 1.5, 2.0];
    let ids = vec![100u64, 200, 300, 400];

    write_id_map(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[], &ids).unwrap();
    let (bw, d, n, p, s, shift, scale_tq, slot_to_id) = load_id_map(&path).unwrap();

    assert_eq!(bw, bit_width);
    assert_eq!(d, dim);
    assert_eq!(n, n_vectors);
    assert_eq!(p, packed);
    assert_eq!(s, scales);
    assert!(shift.is_empty());
    assert!(scale_tq.is_empty());
    assert_eq!(slot_to_id, ids);
    std::fs::remove_file(&path).ok();
}

#[test]
fn tvim_v1_file_is_rejected_with_upgrade_hint() {
    // Hand-construct a turbovec ≤ 0.4.3 `.tvim` file: TVIM magic, version
    // byte = 1, then the same v1 core layout.
    let path = temp_path("v1.tvim");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVIM").unwrap();
        f.write_all(&[1u8]).unwrap(); // version
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(1u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&vec![0u8; (32 / 8) * 4]).unwrap(); // packed codes
        f.write_all(&(1.0f32).to_le_bytes()).unwrap(); // norm
        f.write_all(&(42u64).to_le_bytes()).unwrap(); // id
    }

    let err = load_id_map(&path).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("turbovec ≤ 0.4.3") && msg.contains("Rebuild"),
        "expected upgrade hint, got: {}",
        msg
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_truncated_payload_errors_cleanly() {
    // Write a valid v3 .tv file, then truncate it mid-payload. `load`
    // must surface a clean io::Error (UnexpectedEof) rather than panic
    // or return malformed state.
    let path = temp_path("truncated.tv");
    let bit_width = 4;
    let dim = 32;
    let n_vectors = 5;
    let packed = vec![0xCDu8; (dim / 8) * bit_width * n_vectors];
    let scales = vec![1.0f32; n_vectors];
    write(&path, bit_width, dim, n_vectors, &packed, &scales, &[], &[]).unwrap();

    // Truncate the file to half its size.
    let len = std::fs::metadata(&path).unwrap().len();
    let f = File::options().write(true).open(&path).unwrap();
    f.set_len(len / 2).unwrap();
    drop(f);

    let err = load(&path).unwrap_err();
    assert_eq!(
        err.kind(),
        std::io::ErrorKind::UnexpectedEof,
        "expected UnexpectedEof on truncated file, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_unsupported_version_errors_with_useful_message() {
    // Hand-construct a .tv file with a recognised magic but a version
    // byte we don't support. Loader must surface a clean InvalidData
    // error rather than try to parse with the wrong layout.
    let path = temp_path("future_version.tv");
    let mut f = File::create(&path).unwrap();
    f.write_all(b"TVPI").unwrap();
    f.write_all(&[99u8]).unwrap();  // version=99 — not 2, not 3
    // Pad with arbitrary bytes so the read doesn't fail before the
    // version check.
    f.write_all(&[0u8; 64]).unwrap();
    drop(f);

    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        msg.contains("unsupported"),
        "expected 'unsupported' in error message, got: {msg}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_v3_invalid_n_calib_errors_cleanly() {
    // Hand-construct a v3 .tv file whose n_calib is neither 0 nor dim.
    // Loader must reject with InvalidData per the contract in io.rs.
    let path = temp_path("bad_n_calib.tv");
    let bit_width = 4u8;
    let dim = 32u32;
    let n_vectors = 1u32;

    let mut f = File::create(&path).unwrap();
    f.write_all(b"TVPI").unwrap();
    f.write_all(&[3u8]).unwrap();  // version=3
    f.write_all(&[bit_width]).unwrap();
    f.write_all(&dim.to_le_bytes()).unwrap();
    f.write_all(&n_vectors.to_le_bytes()).unwrap();
    // Packed codes: (dim/8) * bit_width * n_vectors = 4 * 4 * 1 = 16 bytes.
    f.write_all(&[0xAAu8; 16]).unwrap();
    // Scale: 1 f32.
    f.write_all(&1.0f32.to_le_bytes()).unwrap();
    // n_calib = 7 — neither 0 nor dim (32). Invalid.
    f.write_all(&7u32.to_le_bytes()).unwrap();
    // Pad with junk so the read doesn't fail before the n_calib check.
    f.write_all(&[0u8; 64]).unwrap();
    drop(f);

    let err = load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("n_calib"),
        "expected 'n_calib' in error message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_dim_not_multiple_of_8_errors_cleanly() {
    // Reproduces the FFI-abort defect: a header with dim not a multiple of
    // 8 makes the read layer's `(dim/8)*bit_width*n_vectors` formula
    // disagree with `from_parts`'s `n_vectors*dim*bit_width/8`, tripping an
    // assert. The read layer must now reject it as InvalidData first.
    let path = temp_path("bad_dim.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVPI").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(12u32).to_le_bytes()).unwrap(); // dim=12 (not a multiple of 8)
        f.write_all(&(2u32).to_le_bytes()).unwrap(); // n_vectors
        // (dim/8)*bit_width*n_vectors = 1*4*2 = 8 packed bytes.
        f.write_all(&[0u8; 8]).unwrap();
        f.write_all(&[0u8; 8]).unwrap(); // 2 f32 scales
        f.write_all(&0u32.to_le_bytes()).unwrap(); // n_calib = 0
    }
    let err = load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("multiple of 8"),
        "expected 'multiple of 8' in message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_lazy_header_with_nonzero_n_errors_cleanly() {
    // A lazy header (dim=0) must have n_vectors=0; otherwise `from_parts`
    // would assert. Read layer rejects it as InvalidData.
    let path = temp_path("lazy_bad_n.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVPI").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(0u32).to_le_bytes()).unwrap(); // dim=0 (lazy)
        f.write_all(&(3u32).to_le_bytes()).unwrap(); // n_vectors=3 (invalid for lazy)
        // (dim/8)*bit_width*n_vectors = 0 packed bytes.
        f.write_all(&[0u8; 12]).unwrap(); // 3 f32 scales
        f.write_all(&0u32.to_le_bytes()).unwrap(); // n_calib = 0
    }
    let err = load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("n_vectors=0"),
        "expected lazy n_vectors message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_invalid_bit_width_errors_cleanly() {
    // bit_width must be 2, 3, or 4.
    let path = temp_path("bad_bw.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVPI").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[7u8]).unwrap(); // bit_width=7 (invalid)
        f.write_all(&(32u32).to_le_bytes()).unwrap(); // dim
        f.write_all(&(1u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&[0u8; 64]).unwrap(); // padding (read fails after bw check anyway)
    }
    let err = load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("bit_width"),
        "expected 'bit_width' in message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_pathological_sizes_do_not_overflow() {
    // A crafted header with huge dim and n_vectors must error rather than
    // wrap the packed-bytes computation or attempt a giant allocation.
    let path = temp_path("overflow.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVPI").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(0xFFFF_FFF8u32).to_le_bytes()).unwrap(); // dim ~4e9, multiple of 8
        f.write_all(&(0xFFFF_FFFFu32).to_le_bytes()).unwrap(); // n_vectors ~4e9
        f.write_all(&[0u8; 16]).unwrap(); // a little payload
    }
    // The crafted dim/n product is gigantic but still fits in usize, so the
    // plain overflow check passes; the incremental read must cap memory and
    // surface a clean EOF (or InvalidData) instead of aborting on a giant
    // allocation. Reaching this assertion at all proves the process survived.
    let err = load(&path).unwrap_err();
    assert!(
        matches!(
            err.kind(),
            std::io::ErrorKind::UnexpectedEof | std::io::ErrorKind::InvalidData
        ),
        "expected clean error, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_dim_exceeding_max_dim_errors_cleanly() {
    // A ~22-byte crafted header (valid magic, bw=4, dim=1048576, n=0) used
    // to load cleanly — dim is a multiple of 8 and n=0 means no payload —
    // and then abort the process on the dim × dim f64 rotation-matrix
    // allocation at first prepare/search. The read layer must reject any
    // dim above MAX_DIM as InvalidData instead.
    let path = temp_path("huge_dim.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVPI").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(1_048_576u32).to_le_bytes()).unwrap(); // dim=2^20 > MAX_DIM
        f.write_all(&(0u32).to_le_bytes()).unwrap(); // n_vectors=0
        f.write_all(&0u32.to_le_bytes()).unwrap(); // n_calib = 0
    }
    let err = load(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("maximum supported dim"),
        "expected MAX_DIM rejection message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_dim_at_max_dim_is_accepted() {
    // dim == MAX_DIM is the inclusive bound: a header claiming exactly
    // MAX_DIM (with no vectors) must still load.
    let path = temp_path("max_dim.tv");
    write(&path, 4, turbovec::MAX_DIM, 0, &[], &[], &[], &[]).unwrap();
    let (bw, d, n, ..) = load(&path).unwrap();
    assert_eq!(bw, 4);
    assert_eq!(d, turbovec::MAX_DIM);
    assert_eq!(n, 0);
    std::fs::remove_file(&path).ok();
}

#[test]
fn tvim_dim_exceeding_max_dim_errors_cleanly() {
    // Same crafted-header defect, .tvim path.
    let path = temp_path("huge_dim.tvim");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVIM").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(1_048_576u32).to_le_bytes()).unwrap(); // dim=2^20 > MAX_DIM
        f.write_all(&(0u32).to_le_bytes()).unwrap(); // n_vectors=0
        f.write_all(&0u32.to_le_bytes()).unwrap(); // n_calib = 0
    }
    let err = load_id_map(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("maximum supported dim"),
        "expected MAX_DIM rejection message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tvim_dim_not_multiple_of_8_errors_cleanly() {
    // Same defect, .tvim path.
    let path = temp_path("bad_dim.tvim");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"TVIM").unwrap();
        f.write_all(&[3u8]).unwrap(); // version=3
        f.write_all(&[4u8]).unwrap(); // bit_width
        f.write_all(&(12u32).to_le_bytes()).unwrap(); // dim=12 (not a multiple of 8)
        f.write_all(&(2u32).to_le_bytes()).unwrap(); // n_vectors
        f.write_all(&[0u8; 8]).unwrap(); // packed
        f.write_all(&[0u8; 8]).unwrap(); // 2 f32 scales
        f.write_all(&0u32.to_le_bytes()).unwrap(); // n_calib = 0
        f.write_all(&[0u8; 16]).unwrap(); // 2 u64 ids
    }
    let err = load_id_map(&path).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    assert!(
        err.to_string().contains("multiple of 8"),
        "expected 'multiple of 8' in message, got: {err}",
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn tv_garbage_file_rejected_without_upgrade_hint() {
    let path = temp_path("garbage.tv");
    {
        let mut f = File::create(&path).unwrap();
        f.write_all(b"NOPE").unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }
    let err = load(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("wrong magic"), "expected wrong-magic error, got: {}", msg);
    assert!(
        !msg.contains("turbovec ≤ 0.4.3"),
        "should not suggest upgrade for garbage: {}",
        msg
    );
    std::fs::remove_file(&path).ok();
}
