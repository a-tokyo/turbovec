//! napi binding for [`turbovec_core::TurboQuantIndex`].

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::{
    checked_uint_arg, dim_required, index_out_of_range, invalid_query_value, io_error,
    map_add_error, map_construct_error, mask_length_mismatch, query_dim_mismatch, ErrCode,
};

/// Reused from core: any query coordinate with `|value| >= MAX_INPUT_MAGNITUDE`
/// (or that is non-finite) will panic in the core search kernel; we reject it
/// here before crossing the FFI boundary. `MAX_DIM` bounds `dim` the same way
/// the core read layer bounds it for serialized headers.
use turbovec_core::{MAX_DIM, MAX_INPUT_MAGNITUDE};

// We want the proc-macro to see a literal `Result` path segment so it
// sets `is_ret_result = true` and generates the correct match-based
// dispatch.  Using a type alias like `NResult` hides the ident and the
// macro falls back to treating the return value as a plain non-result
// type, causing `ToNapiValue` failures.  We therefore spell out the
// return type as `napi::Result<T, ErrCode>` throughout.

/// Search result returned by `TurboQuantIndex.search`.
#[napi(object)]
pub struct SearchResult {
    /// Flat row-major scores: nq × k.
    pub scores: Float32Array,
    /// Flat row-major slot indices (i64 as BigInt): nq × k.
    pub indices: BigInt64Array,
    /// Number of queries.
    pub nq: u32,
    /// Effective k (may be less than requested k if index is small or mask
    /// restricts the eligible set).
    pub k: u32,
}

/// Options bag for `TurboQuantIndex.search`.
#[napi(object)]
pub struct SearchOptions {
    /// Boolean mask of length `index.length`. Only `true` slots are eligible.
    pub mask: Option<Vec<bool>>,
}

#[napi(js_name = "TurboQuantIndex")]
pub struct TurboQuantIndex {
    inner: turbovec_core::TurboQuantIndex,
}

#[napi]
impl TurboQuantIndex {
    /// Construct a `TurboQuantIndex`.
    ///
    /// - `dim` — vector dimensionality (must be a positive multiple of 8).
    ///   Omit or pass `null`/`undefined` for a lazy index that commits its
    ///   dim on the first `add` call.
    /// - `bitWidth` — quantisation precision: `2`, `3`, or `4` (default `4`).
    #[napi(constructor)]
    pub fn new(
        dim: Option<f64>,
        bit_width: Option<f64>,
    ) -> napi::Result<Self, ErrCode> {
        // Validate at the boundary (see `checked_uint_arg`): napi's raw u32
        // conversion would ToUint32-wrap `-8` into a ~4-billion dim whose
        // dim × dim rotation matrix aborts the process. `bitWidth` only
        // needs to fit a byte here — the core then enforces 2..=4 with
        // BIT_WIDTH_OUT_OF_RANGE.
        let bw = match bit_width {
            Some(b) => checked_uint_arg("bitWidth", b, u8::MAX as usize)?,
            None => 4,
        };
        let inner = match dim {
            Some(d) => {
                let d = checked_uint_arg("dim", d, MAX_DIM)?;
                turbovec_core::TurboQuantIndex::new(d, bw).map_err(map_construct_error)?
            }
            None => turbovec_core::TurboQuantIndex::new_lazy(bw).map_err(map_construct_error)?,
        };
        Ok(Self { inner })
    }

    /// Add vectors to the index.
    ///
    /// `vectors` is a flat row-major `Float32Array` of length `n * dim`.
    /// On a lazy index, `dim` is required on the first call and commits the
    /// index dimensionality.
    #[napi]
    pub fn add(
        &mut self,
        vectors: Float32Array,
        dim: Option<f64>,
    ) -> napi::Result<(), ErrCode> {
        let dim = match dim {
            Some(d) => Some(checked_uint_arg("dim", d, MAX_DIM)?),
            None => None,
        };
        let effective_dim: usize = match self.inner.dim_opt() {
            Some(d) => {
                // If caller explicitly passed a conflicting dim, surface it
                // as DIM_MISMATCH rather than silently ignoring it.
                if let Some(caller_dim) = dim {
                    if caller_dim != d {
                        return Err(crate::error::map_add_error(
                            turbovec_core::AddError::DimMismatch {
                                existing: d,
                                got: caller_dim,
                            },
                        ));
                    }
                }
                d
            }
            None => match dim {
                Some(d) => d,
                None => return Err(dim_required()),
            },
        };

        // Pre-validate buffer length so we never panic in core.
        if effective_dim == 0 || !vectors.len().is_multiple_of(effective_dim) {
            return Err(map_add_error(
                turbovec_core::AddError::VectorBufferNotMultipleOfDim {
                    vectors_len: vectors.len(),
                    dim: effective_dim,
                },
            ));
        }

        // Snapshot the borrowed vector buffer BEFORE passing it to core.
        // A SharedArrayBuffer-backed Float32Array can be mutated by a Worker
        // thread between core's first read (first_invalid_coord validation) and
        // its second read (the actual insert), turning a valid buffer into one
        // with a NaN that panics and aborts the Node process across the FFI
        // boundary. Copying first means both reads see identical bytes; the
        // copy is cheap relative to the quantisation work.
        let owned: Vec<f32> = vectors.to_vec();
        self.inner.add_2d(&owned, effective_dim).map_err(map_add_error)
    }

    /// Run a top-`k` search against the index.
    ///
    /// `queries` is a flat row-major `Float32Array` of length `nq * dim`.
    /// The optional `mask` is a `boolean[]` of length equal to `this.length`
    /// that restricts the candidates to `true` slots.
    #[napi]
    pub fn search(
        &self,
        queries: Float32Array,
        k: f64,
        options: Option<SearchOptions>,
    ) -> napi::Result<SearchResult, ErrCode> {
        let k = checked_uint_arg("k", k, u32::MAX as usize)?;

        // Snapshot before FFI — SAB TOCTOU guard (see the first occurrence in add).
        let queries_owned: Vec<f32> = queries.to_vec();
        let q_slice: &[f32] = &queries_owned;

        // Derive nq — requires a committed dim for non-empty queries.
        let nq: usize = if q_slice.is_empty() {
            0
        } else {
            match self.inner.dim_opt() {
                Some(dim) => {
                    if !q_slice.len().is_multiple_of(dim) {
                        // Report the raw buffer length as "got" so the message
                        // reads "query buffer length X is not a multiple of
                        // index dim Y".  The helper message already says
                        // "query dim {got} does not match index dim {expected}",
                        // which is clear enough for the non-multiple case.
                        return Err(query_dim_mismatch(q_slice.len(), dim));
                    }
                    q_slice.len() / dim
                }
                None => return Err(dim_required()),
            }
        };

        // Pre-validate query coordinates — reject non-finite and huge-magnitude
        // values before they cross the FFI boundary and panic in the core.
        if !q_slice.is_empty() {
            if let Some(dim) = self.inner.dim_opt() {
                for (i, &x) in q_slice.iter().enumerate() {
                    if !x.is_finite() || x.abs() >= MAX_INPUT_MAGNITUDE {
                        let query_index = i / dim;
                        let coord_index = i % dim;
                        return Err(invalid_query_value(query_index, coord_index, x));
                    }
                }
            }
        }

        // Validate and extract mask.
        let mask_vec: Option<Vec<bool>> = match options.and_then(|o| o.mask) {
            Some(m) => {
                let expected = self.inner.len();
                if m.len() != expected {
                    return Err(mask_length_mismatch(m.len(), expected));
                }
                Some(m)
            }
            None => None,
        };

        let results = self
            .inner
            .search_with_mask(q_slice, k, mask_vec.as_deref());

        let effective_k = results.k;
        let indices_data: Vec<i64> = results.indices;

        Ok(SearchResult {
            scores: results.scores.into(),
            indices: indices_data.into(),
            nq: nq as u32,
            k: effective_k as u32,
        })
    }

    /// Remove the vector at `idx` in O(1) by swapping with the last vector.
    ///
    /// Returns the old index of the moved vector.
    /// Throws `INDEX_OUT_OF_RANGE` if `idx >= this.length`.
    #[napi(js_name = "swapRemove")]
    pub fn swap_remove(&mut self, idx: f64) -> napi::Result<u32, ErrCode> {
        let idx = checked_uint_arg("idx", idx, u32::MAX as usize)?;
        let len = self.inner.len();
        if idx >= len {
            return Err(index_out_of_range(idx, len));
        }
        Ok(self.inner.swap_remove(idx) as u32)
    }

    /// Warm up search caches so the first `search` call does not pay the
    /// one-time initialisation cost.
    #[napi]
    pub fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialise the index to `path`.
    #[napi]
    pub fn write(&self, path: String) -> napi::Result<(), ErrCode> {
        self.inner.write(&path).map_err(|e| io_error(&e))
    }

    /// Load an index from `path`.
    ///
    /// Rejects any serialized index whose committed `dim` exceeds
    /// [`turbovec_core::MAX_DIM`]: a crafted header can claim a
    /// huge-but-multiple-of-8 dim that loads cleanly from the core read
    /// layer and then aborts the Node process on the `dim × dim` f64
    /// rotation-matrix allocation at the first `search`/`prepare` call.
    #[napi(factory)]
    pub fn load(path: String) -> napi::Result<Self, ErrCode> {
        let inner =
            turbovec_core::TurboQuantIndex::load(&path).map_err(|e| io_error(&e))?;
        if let Some(dim) = inner.dim_opt() {
            if dim > MAX_DIM {
                return Err(io_error(&std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "index dim {dim} exceeds the maximum supported dim {MAX_DIM}; \
                         the file may be malformed or crafted"
                    ),
                )));
            }
        }
        Ok(Self { inner })
    }

    /// Number of vectors in the index.
    #[napi(getter)]
    pub fn length(&self) -> u32 {
        self.inner.len() as u32
    }

    /// Vector dimensionality, or `null` for a lazy uncommitted index.
    #[napi(getter)]
    pub fn dim(&self) -> Option<u32> {
        self.inner.dim_opt().map(|d| d as u32)
    }

    /// Quantisation bit-width (2, 3, or 4).
    #[napi(getter, js_name = "bitWidth")]
    pub fn bit_width(&self) -> u32 {
        self.inner.bit_width() as u32
    }
}
