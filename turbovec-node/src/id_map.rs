//! napi binding for [`turbovec_core::IdMapIndex`].

use napi::bindgen_prelude::*;
use napi_derive::napi;

use crate::error::{
    allowlist_empty, allowlist_unknown_ids, dim_required, invalid_query_value, io_error,
    map_add_error, map_construct_error, query_dim_mismatch, ErrCode,
};

/// Reused from core to reject invalid input before the FFI crossing.
use turbovec_core::MAX_INPUT_MAGNITUDE;

/// Search result returned by `IdMapIndex.search`.
#[napi(object)]
pub struct IdSearchResult {
    /// Flat row-major scores: nq × k.
    pub scores: Float32Array,
    /// Flat row-major external ids (u64 as BigInt): nq × k.
    pub ids: BigUint64Array,
    /// Number of queries.
    pub nq: u32,
    /// Effective k.
    pub k: u32,
}

/// Options bag for `IdMapIndex.search`.
#[napi(object)]
pub struct IdSearchOptions {
    /// Restrict results to these external ids.
    pub allowlist: Option<BigUint64Array>,
}

#[napi(js_name = "IdMapIndex")]
pub struct IdMapIndex {
    inner: turbovec_core::IdMapIndex,
}

#[napi]
impl IdMapIndex {
    /// Construct an `IdMapIndex`.
    ///
    /// Same `dim`/`bitWidth` semantics as `TurboQuantIndex`.
    #[napi(constructor)]
    pub fn new(dim: Option<u32>, bit_width: Option<u32>) -> napi::Result<Self, ErrCode> {
        let bw = bit_width.unwrap_or(4) as usize;
        let inner = match dim {
            Some(d) => {
                turbovec_core::IdMapIndex::new(d as usize, bw).map_err(map_construct_error)?
            }
            None => turbovec_core::IdMapIndex::new_lazy(bw).map_err(map_construct_error)?,
        };
        Ok(Self { inner })
    }

    /// Add vectors with stable external ids.
    ///
    /// `vectors` — flat row-major `Float32Array`.
    /// `ids`     — `BigUint64Array` with the same element count as the
    ///             number of rows in `vectors` (`vectors.length / dim`).
    /// `dim`     — required when the index is still lazy.
    #[napi(js_name = "addWithIds")]
    pub fn add_with_ids(
        &mut self,
        vectors: Float32Array,
        ids: BigUint64Array,
        dim: Option<u32>,
    ) -> napi::Result<(), ErrCode> {
        let effective_dim: usize = match self.inner.dim_opt() {
            Some(d) => {
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

        // Pre-validate buffer length.
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

        let v_slice: &[f32] = &vectors;
        let i_slice: &[u64] = &ids;

        self.inner
            .add_with_ids_2d(v_slice, effective_dim, i_slice)
            .map_err(map_add_error)
    }

    /// Remove the vector with the given external id.
    ///
    /// Returns `true` if the id was present, `false` otherwise.
    /// Negative BigInts and values that exceed u64 are definitively absent —
    /// they cannot alias any stored u64 id, so `remove` returns `false`
    /// without touching the index.
    #[napi]
    pub fn remove(&mut self, id: BigInt) -> bool {
        let (sign_bit, id_u64, lossless) = id.get_u64();
        if sign_bit || !lossless {
            // Negative or out-of-u64-range — cannot be a stored id.
            return false;
        }
        self.inner.remove(id_u64)
    }

    /// Test whether the index contains the given external id.
    /// Negative BigInts and values that exceed u64 are definitively absent.
    #[napi]
    pub fn contains(&self, id: BigInt) -> bool {
        let (sign_bit, id_u64, lossless) = id.get_u64();
        if sign_bit || !lossless {
            return false;
        }
        self.inner.contains(id_u64)
    }

    /// Search for the top-`k` nearest ids for each query.
    ///
    /// Optional `allowlist` restricts the result set to those external ids.
    #[napi]
    pub fn search(
        &self,
        queries: Float32Array,
        k: u32,
        options: Option<IdSearchOptions>,
    ) -> napi::Result<IdSearchResult, ErrCode> {
        let k = k as usize;
        let q_slice: &[f32] = &queries;

        // Derive nq.
        let nq: usize = if q_slice.is_empty() {
            0
        } else {
            match self.inner.dim_opt() {
                Some(dim) => {
                    if !q_slice.len().is_multiple_of(dim) {
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

        // Extract and validate allowlist.
        // `allow_owned` keeps the Vec<u64> alive for the duration of the
        // `search_with_allowlist` call below.
        let allow_owned: Option<Vec<u64>> = match options.and_then(|o| o.allowlist) {
            Some(al) => {
                let slice_ref: &[u64] = &al;
                if slice_ref.is_empty() {
                    return Err(allowlist_empty());
                }

                // Collect up to 6 unknown ids for the error message.
                let mut unknown: Vec<u64> = Vec::new();
                for &id in slice_ref {
                    if !self.inner.contains(id) {
                        unknown.push(id);
                        if unknown.len() >= 6 {
                            break;
                        }
                    }
                }
                if !unknown.is_empty() {
                    let more = unknown.len() > 5;
                    let preview: Vec<u64> = unknown.into_iter().take(5).collect();
                    return Err(allowlist_unknown_ids(&preview, more));
                }

                Some(slice_ref.to_vec())
            }
            None => None,
        };
        let allow_slice: Option<&[u64]> = allow_owned.as_deref();

        let (scores, ids) = self.inner.search_with_allowlist(q_slice, k, allow_slice);

        // Compute effective_k — mirror Python lib.rs lines ~291-303.
        let effective_k = if nq == 0 {
            let n_allowed = match allow_slice {
                Some(s) => {
                    let mut seen = std::collections::HashSet::with_capacity(s.len());
                    s.iter().filter(|id| seen.insert(**id)).count()
                }
                None => self.inner.len(),
            };
            k.min(self.inner.len()).min(n_allowed)
        } else {
            // nq > 0 is guaranteed by the branch above; checked_div avoids
            // the `manual-checked-ops` clippy lint.
            scores.len().checked_div(nq).unwrap_or(0)
        };

        Ok(IdSearchResult {
            scores: scores.into(),
            ids: ids.into(),
            nq: nq as u32,
            k: effective_k as u32,
        })
    }

    /// Warm up search caches.
    #[napi]
    pub fn prepare(&self) {
        self.inner.prepare();
    }

    /// Serialise to a `.tvim` file.
    #[napi]
    pub fn write(&self, path: String) -> napi::Result<(), ErrCode> {
        self.inner.write(&path).map_err(|e| io_error(&e))
    }

    /// Load an `IdMapIndex` from a `.tvim` file.
    #[napi(factory)]
    pub fn load(path: String) -> napi::Result<Self, ErrCode> {
        let inner = turbovec_core::IdMapIndex::load(&path).map_err(|e| io_error(&e))?;
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
