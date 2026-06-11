//! Typed-error layer.
//!
//! `napi::Error<S>` accepts any `S: AsRef<str>` as the status/code type.
//! The napi-rs runtime maps that string to the JavaScript `.code` property
//! on the thrown `Error` object, so callers can do `err.code === "DIM_MISMATCH"`.

use napi::bindgen_prelude::*;
use turbovec_core::{AddError, ConstructError};

/// A string-backed error code that becomes `err.code` in JavaScript.
#[derive(Clone, Debug)]
pub struct ErrCode(pub &'static str);

impl AsRef<str> for ErrCode {
    fn as_ref(&self) -> &str {
        self.0
    }
}

impl From<Status> for ErrCode {
    fn from(_: Status) -> Self {
        ErrCode("GENERIC_FAILURE")
    }
}

fn err(code: &'static str, msg: impl Into<String>) -> napi::Error<ErrCode> {
    napi::Error::new(ErrCode(code), msg.into())
}

// ── AddError → Error<ErrCode> ─────────────────────────────────────────────

pub fn map_add_error(e: AddError) -> napi::Error<ErrCode> {
    match e {
        AddError::DimMismatch { existing, got } => err(
            "DIM_MISMATCH",
            format!("dim mismatch: index dim={existing}, batch dim={got}"),
        ),
        AddError::DimNotMultipleOf8(dim) => err(
            "DIM_NOT_MULTIPLE_OF_8",
            format!("dim must be a multiple of 8, got {dim}"),
        ),
        AddError::VectorBufferNotMultipleOfDim { vectors_len, dim } => err(
            "VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM",
            format!(
                "vector buffer length {vectors_len} not a multiple of dim {dim}"
            ),
        ),
        AddError::IdsCountMismatch { expected, got } => err(
            "IDS_COUNT_MISMATCH",
            format!("expected {expected} ids, got {got}"),
        ),
        AddError::IdAlreadyPresent(id) => err(
            "ID_ALREADY_PRESENT",
            format!("id {id} already present in index"),
        ),
        AddError::InvalidInputValue {
            vector_index,
            coord_index,
            value,
        } => err(
            "INVALID_INPUT_VALUE",
            format!(
                "invalid input value at vector {vector_index}, coord {coord_index}: {value} \
                 (must be finite and |value| < 1e16 to avoid f32 norm overflow)"
            ),
        ),
    }
}

// ── ConstructError → Error<ErrCode> ──────────────────────────────────────

pub fn map_construct_error(e: ConstructError) -> napi::Error<ErrCode> {
    match e {
        ConstructError::BitWidthOutOfRange(bw) => err(
            "BIT_WIDTH_OUT_OF_RANGE",
            format!("bit_width must be 2, 3, or 4, got {bw}"),
        ),
        ConstructError::DimNotPositiveMultipleOf8(dim) => err(
            "DIM_NOT_POSITIVE_MULTIPLE_OF_8",
            format!("dim must be a positive multiple of 8, got {dim}"),
        ),
    }
}

// ── Binding-layer helpers (pre-validation) ────────────────────────────────

pub fn dim_required() -> napi::Error<ErrCode> {
    err(
        "DIM_REQUIRED",
        "index is lazy and no dim has been committed yet; \
         pass a dim argument or construct with TurboQuantIndex(dim, bitWidth)",
    )
}

pub fn query_dim_mismatch(got: usize, expected: usize) -> napi::Error<ErrCode> {
    err(
        "QUERY_DIM_MISMATCH",
        format!("query dim {got} does not match index dim {expected}"),
    )
}

pub fn mask_length_mismatch(got: usize, expected: usize) -> napi::Error<ErrCode> {
    err(
        "MASK_LENGTH_MISMATCH",
        format!("mask length {got} does not match index size {expected}"),
    )
}

pub fn allowlist_empty() -> napi::Error<ErrCode> {
    err("ALLOWLIST_EMPTY", "allowlist is empty")
}

pub fn allowlist_unknown_ids(ids: &[u64], more: bool) -> napi::Error<ErrCode> {
    let suffix = if more { ", ..." } else { "" };
    err(
        "ALLOWLIST_UNKNOWN_ID",
        format!(
            "allowlist contains id(s) not present in index: {ids:?}{suffix}"
        ),
    )
}

pub fn index_out_of_range(idx: usize, len: usize) -> napi::Error<ErrCode> {
    err(
        "INDEX_OUT_OF_RANGE",
        format!("index {idx} out of range for index of length {len}"),
    )
}

pub fn io_error(e: &std::io::Error) -> napi::Error<ErrCode> {
    err("IO_ERROR", format!("{e}"))
}

pub fn invalid_query_value(
    query_index: usize,
    coord_index: usize,
    value: f32,
) -> napi::Error<ErrCode> {
    err(
        "INVALID_INPUT_VALUE",
        format!(
            "invalid query value at query {query_index}, coord {coord_index}: {value} \
             (must be finite and |value| < 1e16 to avoid f32 overflow)"
        ),
    )
}
