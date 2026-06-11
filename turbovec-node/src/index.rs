//! napi binding for [`turbovec_core::TurboQuantIndex`].

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::{
    dim_required, index_out_of_range, invalid_query_value, io_error, map_add_error,
    map_construct_error, mask_length_mismatch, query_dim_mismatch, ErrCode,
};

/// Reused from core: any query coordinate with `|value| >= MAX_INPUT_MAGNITUDE`
/// (or that is non-finite) will panic in the core search kernel; we reject it
/// here before crossing the FFI boundary.
use turbovec_core::MAX_INPUT_MAGNITUDE;

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
        dim: Option<u32>,
        bit_width: Option<u32>,
    ) -> napi::Result<Self, ErrCode> {
        let bw = bit_width.unwrap_or(4) as usize;
        let inner = match dim {
            Some(d) => {
                turbovec_core::TurboQuantIndex::new(d as usize, bw)
                    .map_err(map_construct_error)?
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
        dim: Option<u32>,
    ) -> napi::Result<(), ErrCode> {
        let effective_dim: usize = match self.inner.dim_opt() {
            Some(d) => {
                // If caller explicitly passed a conflicting dim, surface it
                // as DIM_MISMATCH rather than silently ignoring it.
                if let Some(caller_dim) = dim {
                    let caller_dim = caller_dim as usize;
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
                Some(d) => d as usize,
                None => return Err(dim_required()),
            },
        };

        // Pre-validate buffer length so we never panic in core.
        if effective_dim == 0 || !vectors.len().is_multiple_of(effective_dim) {
            return Err(napi::Error::new(
                ErrCode("VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM"),
                format!(
                    "vector buffer length {} not a multiple of dim {}",
                    vectors.len(),
                    effective_dim
                ),
            ));
        }

        let slice: &[f32] = &vectors;
        self.inner.add_2d(slice, effective_dim).map_err(map_add_error)
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
        k: u32,
        options: Option<SearchOptions>,
    ) -> napi::Result<SearchResult, ErrCode> {
        let k = k as usize;
        let q_slice: &[f32] = &queries;

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
    pub fn swap_remove(&mut self, idx: u32) -> napi::Result<u32, ErrCode> {
        let idx = idx as usize;
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
    #[napi(factory)]
    pub fn load(path: String) -> napi::Result<Self, ErrCode> {
        let inner =
            turbovec_core::TurboQuantIndex::load(&path).map_err(|e| io_error(&e))?;
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
