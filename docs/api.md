# API Reference

turbovec exposes two index types and one serialization format per type.

- [`TurboQuantIndex`](#turboquantindex) — positional index, O(1) `swap_remove` delete.
- [`IdMapIndex`](#idmapindex) — stable external `u64` ids on top of `TurboQuantIndex`.
- [File formats](#file-formats) — `.tv` and `.tvim`.

The default examples below are Python. The Rust API mirrors it — see each type's rustdoc for the exact signatures. The Node.js (JavaScript) API is documented in a labelled subsection under each type, plus a [JS error-code table](#javascript-error-codes); it differs in a few places (flat typed-array buffers, `bigint` ids, an explicit-`dim` requirement on the first lazy add) that are called out there.

---

## `TurboQuantIndex`

Positional index. Each vector is identified by its insertion slot (`0..n`). Fast and small, but external references to slots are invalidated by `swap_remove`. If you need stable ids, use [`IdMapIndex`](#idmapindex).

```python
from turbovec import TurboQuantIndex

idx = TurboQuantIndex(dim=1536, bit_width=4)
idx.add(vectors)                        # np.ndarray of shape (n, dim), float32
scores, indices = idx.search(queries, k=10)

idx.swap_remove(5)                      # O(1); the previously-last vector moves into slot 5

idx.write("index.tv")                   # .tv format
loaded = TurboQuantIndex.load("index.tv")
```

`dim` is optional. Omit it to let the index pick up the dimensionality from the first batch of vectors:

```python
idx = TurboQuantIndex(bit_width=4)      # dim inferred on first add
idx.add(vectors)                         # locks dim to vectors.shape[1]
```

Before the first add, `idx.dim` is `None`, `len(idx)` is `0`, and `search()` returns empty results.

### Methods

| Method                                   | Notes                                                                                                                                                                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TurboQuantIndex(dim=None, bit_width=4)` | `bit_width ∈ {2, 3, 4}`. `dim` is optional; when omitted it is inferred from the first `add` call.                                                                                                                                                      |
| `add(vectors)`                           | `vectors` is a contiguous float32 array of shape `(n, dim)`. On a lazy index the first call locks `dim`; subsequent calls must match. Raises `ValueError` on dim mismatch.                                                                              |
| `search(queries, k, *, mask=None)`       | Returns `(scores, indices)`, both shape `(nq, effective_k)`. Indices are `int64` slot positions. `mask` is an optional `bool` array of length `len(idx)`; when given, only slots with `mask[i] == True` contribute. `effective_k = min(k, mask.sum())`. |
| `swap_remove(idx)`                       | O(1). Moves the last vector into `idx`; returns the previous position of that moved vector (so external refs can be updated if needed).                                                                                                                 |
| `prepare()`                              | Optional. Eagerly builds the rotation matrix, Lloyd-Max centroids and SIMD-blocked layout so the first `search` call doesn't pay the one-time cost. No-op on a lazy index that hasn't seen its first add.                                               |
| `write(path)` / `load(path)`             | `.tv` format.                                                                                                                                                                                                                                           |
| `len(idx)` / `idx.dim` / `idx.bit_width` | Introspection. `idx.dim` returns `int` once committed, or `None` on a lazy index that hasn't seen its first add.                                                                                                                                        |

### `swap_remove` semantics

`swap_remove(i)` is named to match Rust's [`Vec::swap_remove`](https://doc.rust-lang.org/std/vec/struct.Vec.html#method.swap_remove): the last element moves into slot `i`, and the vector is truncated by one. It is **not** a shift (FAISS's `IndexPQ::remove_ids` behaviour). Order is not preserved; slot indices of vectors you didn't delete may now point at different vectors than before.

Use [`IdMapIndex`](#idmapindex) if external references have to stay stable across deletes.

### JavaScript (Node.js)

Same semantics as Python, with these JS-specific contracts:

- **Vectors are a flat row-major `Float32Array`**, not a 2-D array. A batch of `n` rows is one `Float32Array` of length `n * dim`; the row count is `length / dim`.
- **Lazy index requires `dim` on the first add.** Unlike Python (which infers `dim` from the numpy array's shape), there is no 2-D shape to read, so the first `add` on a lazy index must pass `dim` explicitly or it throws `err.code === 'DIM_REQUIRED'`. Once committed, subsequent adds may omit it.
- **Search returns flat typed arrays plus `{ nq, k }`** instead of a 2-D tuple. `scores` is a `Float32Array` of length `nq * k`; `indices` is a `BigInt64Array` of slot positions (i64 as `bigint`). Row `i` is `result.scores.slice(i*k, (i+1)*k)` (and likewise for `indices`).

> **Flat-buffer dim note — `add`.** `add(vectors)` ingests exactly `vectors.length / dim` rows. A buffer whose length is not a multiple of `dim` throws `VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM`, but a buffer that _is_ a multiple of a _different_ intended dim is silently taken as that many rows with no error. For example, `add(new Float32Array(16))` on an 8-dim index adds **2 zero-vectors**, not one 16-dim vector. To catch dimension mismatches explicitly, pass the optional `dim` argument — `add(vectors, 16)` on an 8-dim index throws `DIM_MISMATCH`.

> **Flat-buffer dim note — `search`.** The query buffer length must equal `nq * dim`. A 16-float query against an 8-dim index is interpreted as **2 separate queries** (`nq === 2`), not an error. `QUERY_DIM_MISMATCH` only fires when the buffer length is _not_ a multiple of `dim`. Always verify `index.dim` when querying with vectors from a different embedding model.

```js
const { TurboQuantIndex } = require("turbovec");

const idx = new TurboQuantIndex(8, 4); // dim must be a positive multiple of 8

// Two 8-d vectors as one flat row-major Float32Array (length n*dim = 2*8).
const vectors = new Float32Array([
  1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
]);
idx.add(vectors); // row count = vectors.length / dim = 2

const queries = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]); // one 8-d query
const { scores, indices, nq, k } = idx.search(queries, 10);
// scores: Float32Array(nq*k); indices: BigInt64Array(nq*k) of slot positions
// here k = min(10, 2) = 2, nq = 1

const moved = idx.swapRemove(0); // O(1); returns prev position of moved vector

idx.write("index.tv");
const loaded = TurboQuantIndex.load("index.tv");
```

Lazy construction:

```js
const idx = new TurboQuantIndex(); // dim deferred
idx.add(vectors, 8); // dim REQUIRED on first add (else DIM_REQUIRED)
```

Filtered search uses a `boolean[]` mask of length `idx.length` via the options bag:

```js
const mask = new Array(idx.length).fill(true);
mask[disabledSlot] = false;
const { scores, indices, k } = idx.search(queries, 10, { mask });
// k = min(requestedK, number of true mask slots)
```

| Member                                  | Notes                                                                                                                   |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `new TurboQuantIndex(dim?, bitWidth?)`  | `bitWidth ∈ {2,3,4}` (default 4). `dim` optional (positive multiple of 8, ≤ `MAX_DIM` = 65 536); omit for a lazy index. |
| `add(vectors, dim?)`                    | `vectors`: flat row-major `Float32Array` of length `n*dim`. `dim` required on the first add of a lazy index.            |
| `search(queries, k, { mask? })`         | Returns `{ scores: Float32Array, indices: BigInt64Array, nq, k }`. `mask` is a `boolean[]` of length `this.length`.     |
| `swapRemove(idx)`                       | O(1); returns the prior slot of the moved vector. Throws `INDEX_OUT_OF_RANGE` if `idx >= this.length`.                  |
| `prepare()`                             | Warm up search caches.                                                                                                  |
| `write(path)` / `static load(path)`     | `.tv` format.                                                                                                           |
| `length` / `dim` / `bitWidth` (getters) | `dim` is `number` once committed, `null` on a lazy uncommitted index.                                                   |

---

## `IdMapIndex`

Stable-id wrapper around `TurboQuantIndex`. Roughly equivalent to FAISS's `IndexIDMap2` — hash-table backed, O(1) `remove(id)`.

```python
import numpy as np
from turbovec import IdMapIndex

idx = IdMapIndex(dim=1536, bit_width=4)
idx.add_with_ids(vectors, np.array([1001, 1002, 1003], dtype=np.uint64))

scores, ids = idx.search(queries, k=10)   # ids are uint64 external ids

idx.remove(1002)                           # O(1) by id
assert 1003 in idx                         # __contains__ sugar

idx.write("index.tvim")                    # .tvim format
loaded = IdMapIndex.load("index.tvim")
```

As with [`TurboQuantIndex`](#turboquantindex), `dim` is optional and gets inferred from the first `add_with_ids` call:

```python
idx = IdMapIndex(bit_width=4)            # dim inferred on first add
idx.add_with_ids(vectors, ids)           # locks dim to vectors.shape[1]
```

### Methods

| Method                                                 | Notes                                                                                                                                                                                                                                                                                               |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `IdMapIndex(dim=None, bit_width=4)`                    | `dim` is optional; when omitted it is inferred from the first `add_with_ids` call.                                                                                                                                                                                                                  |
| `add_with_ids(vectors, ids)`                           | `ids` is a `uint64` array with length `vectors.shape[0]`. On a lazy index the first call locks `dim`. Raises `ValueError` on dim mismatch, duplicate ids, or `len(ids) != vectors.shape[0]`.                                                                                                        |
| `remove(id) -> bool`                                   | `True` if the id was present and removed, `False` otherwise. O(1).                                                                                                                                                                                                                                  |
| `search(queries, k, *, allowlist=None)`                | Returns `(scores, ids)` — `ids` are `uint64` external ids. `allowlist` is an optional `uint64` array of ids; when given, results are restricted to those ids and `effective_k = min(k, len(allowlist) after de-duplication)`. Raises `ValueError` on empty allowlist and `KeyError` on unknown ids. |
| `contains(id)` / `id in idx`                           | Membership.                                                                                                                                                                                                                                                                                         |
| `write(path)` / `load(path)`                           | `.tvim` format.                                                                                                                                                                                                                                                                                     |
| `len(idx)` / `idx.dim` / `idx.bit_width` / `prepare()` | Same as `TurboQuantIndex`.                                                                                                                                                                                                                                                                          |

### When to use which

- `TurboQuantIndex` — you never delete, or you're fine with positional ids.
- `IdMapIndex` — you need stable external ids (e.g. string-id → vector mapping maintained by the caller).

All the framework integrations (LangChain, LlamaIndex, Haystack) use `IdMapIndex` internally for exactly this reason.

### JavaScript (Node.js)

Same JS contracts as [`TurboQuantIndex`](#javascript-nodejs) (flat `Float32Array` vectors, `{ nq, k }` result shape, explicit `dim` on first lazy add), plus:

- **Ids are `bigint`**, passed as a `BigUint64Array` (matching numpy `uint64` in Python). `addWithIds(vectors, ids, dim?)` takes a `BigUint64Array` whose element count equals the row count (`vectors.length / dim`).
- **Search returns `ids` (a `BigUint64Array`)** instead of `indices` — your external `u64` ids as `bigint`. Otherwise the result is `{ scores, ids, nq, k }` with the same flat row-major layout.
- **`remove(id)` / `contains(id)` take a `bigint`.** Negative `bigint`s and values that exceed `u64` are definitively absent (they cannot alias any stored id), so they return `false` without touching the index.

```js
const { IdMapIndex } = require("turbovec");

const idx = new IdMapIndex(8, 4);

// Two 8-d vectors as one flat row-major Float32Array (length n*dim = 2*8).
const vectors = new Float32Array([
  1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
]);
const ids = BigUint64Array.from([1001n, 1002n]); // one bigint id per row
idx.addWithIds(vectors, ids);

const queries = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]); // one 8-d query
const { scores, ids: hitIds, nq, k } = idx.search(queries, 10); // hitIds: BigUint64Array

idx.remove(1002n); // O(1); returns true if present
idx.contains(1001n); // boolean

idx.write("index.tvim");
const loaded = IdMapIndex.load("index.tvim");
```

Filtered search uses an `allowlist` of external ids (`BigUint64Array`):

```js
const allowed = BigUint64Array.from([1003n, 1010n, 1042n]);
const { scores, ids, k } = idx.search(queries, 10, { allowlist: allowed });
// k = min(requestedK, allowed.length after de-duplication); throws ALLOWLIST_EMPTY / ALLOWLIST_UNKNOWN_ID on bad input
```

| Member                                            | Notes                                                                                                                              |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `new IdMapIndex(dim?, bitWidth?)`                 | Same `dim`/`bitWidth` semantics as `TurboQuantIndex` (positive multiple of 8, ≤ `MAX_DIM` = 65 536).                               |
| `addWithIds(vectors, ids, dim?)`                  | `vectors`: flat `Float32Array`. `ids`: `BigUint64Array`, element count = `vectors.length / dim`. `dim` required on first lazy add. |
| `search(queries, k, { allowlist? })`              | Returns `{ scores: Float32Array, ids: BigUint64Array, nq, k }`. `allowlist` is a `BigUint64Array` of external ids.                 |
| `remove(id)` → `boolean`                          | `id: bigint`. `true` if present and removed. O(1).                                                                                 |
| `contains(id)` → `boolean`                        | `id: bigint`. Membership test.                                                                                                     |
| `prepare()` / `write(path)` / `static load(path)` | `.tvim` format.                                                                                                                    |
| `length` / `dim` / `bitWidth` (getters)           | Same as `TurboQuantIndex`.                                                                                                         |

### JavaScript error codes

Errors thrown by the native layer carry a stable string `err.code` so callers can branch with `err.code === 'DIM_MISMATCH'`. The full set:

| `err.code`                          | Raised when                                                                                                                                                                                                       |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DIM_MISMATCH`                      | Add batch dim differs from the index's committed dim.                                                                                                                                                             |
| `DIM_NOT_MULTIPLE_OF_8`             | The dim committed on a lazy add is not a multiple of 8.                                                                                                                                                           |
| `VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM` | `vectors.length` is not a multiple of `dim`.                                                                                                                                                                      |
| `IDS_COUNT_MISMATCH`                | `ids.length` ≠ the number of vector rows (`vectors.length / dim`).                                                                                                                                                |
| `ID_ALREADY_PRESENT`                | An id passed to `addWithIds` already exists in the index.                                                                                                                                                         |
| `INVALID_INPUT_VALUE`               | A vector or query coordinate is non-finite (NaN/±Inf) or `\|value\| >= 1e16`.                                                                                                                                      |
| `BIT_WIDTH_OUT_OF_RANGE`            | Constructor `bitWidth` is not 2, 3, or 4.                                                                                                                                                                         |
| `DIM_NOT_POSITIVE_MULTIPLE_OF_8`    | Constructor `dim` is not a positive multiple of 8.                                                                                                                                                                |
| `QUERY_DIM_MISMATCH`                | A search query's dim differs from the index dim.                                                                                                                                                                  |
| `MASK_LENGTH_MISMATCH`              | `mask.length` ≠ `index.length`.                                                                                                                                                                                   |
| `ALLOWLIST_EMPTY`                   | An empty `allowlist` was supplied to `search`.                                                                                                                                                                    |
| `ALLOWLIST_UNKNOWN_ID`              | An `allowlist` id is not present in the index.                                                                                                                                                                    |
| `INDEX_OUT_OF_RANGE`                | `swapRemove(idx)` called with `idx >= index.length`.                                                                                                                                                              |
| `DIM_REQUIRED`                      | First add on a lazy index without a `dim` argument; or a non-empty `search` on a lazy index before any `add` has committed a dim. Note: Python returns an empty `(nq, 0)` result in this case; JS throws instead. |
| `IO_ERROR`                          | A `write` / `load` filesystem or deserialization error.                                                                                                                                                           |
| `INVALID_ARGUMENT`                  | A numeric constructor or method argument is negative, fractional, non-finite, or exceeds the allowed range (e.g. `dim > MAX_DIM`, `k < 0`). Python raises `OverflowError` in equivalent cases.                    |
| `GENERIC_FAILURE`                   | Internal napi-rs runtime failure (e.g. allocation failure); not normally reachable from user code.                                                                                                                |

---

## Filtering

Both index types support restricting the returned top-`k` to a caller-supplied subset of vectors. Unlike post-filtering (search then drop), the kernel never inserts disallowed vectors into the per-query heap, so you always get up to `k` results from the allowed set rather than fewer.

```python
# IdMapIndex — allowlist of external ids (typical use)
allowed = np.array([1003, 1010, 1042], dtype=np.uint64)
scores, ids = idx.search(queries, k=10, allowlist=allowed)
# scores.shape == (nq, min(k, len(allowed))) == (nq, 3)

# TurboQuantIndex — bool mask over slots
mask = np.ones(len(idx), dtype=bool)
mask[disabled_slots] = False
scores, slots = idx.search(queries, k=10, mask=mask)
```

The output shape is `(nq, min(k, n_allowed))` — same shrinking behaviour you already see when `k > len(idx)`. No `-1` / `NaN` padding; pad on the caller side if you need a fixed-width batch.

Common use cases:

- Hybrid retrieval where a SQL/BM25 stage produces a candidate id set.
- Access control or multi-tenant queries (only return ids the caller can see).
- Time-windowed search (e.g. only documents from the last 7 days).

---

## File formats

### `.tv` — `TurboQuantIndex`

```
┌──────────────────────────────────────┐
│ magic    "TVPI"  (4 bytes)            │
│ version  u8    = 3                     │
├──────────────────────────────────────┤
│ core header                           │
│   bit_width  (u8)                     │
│   dim        (u32 LE)                 │
│   n_vectors  (u32 LE)                 │
├──────────────────────────────────────┤
│ packed codes                          │
│   (dim / 8) * bit_width * n_vectors   │
├──────────────────────────────────────┤
│ scales  (n_vectors × f32 LE)          │
│   per-vector length-renormalization   │
├──────────────────────────────────────┤
│ TQ+ trailer                           │
│   n_calib  (u32 LE)  — 0 or dim       │
│   shift    (n_calib × f32 LE)         │
│   scale    (n_calib × f32 LE)         │
└──────────────────────────────────────┘
```

### `.tvim` — `IdMapIndex`

```
┌──────────────────────────────────────┐
│ magic    "TVIM"  (4 bytes)            │
│ version  u8    = 3                     │
├──────────────────────────────────────┤
│ core payload (same as .tv:            │
│   header + codes + scales + TQ+)      │
├──────────────────────────────────────┤
│ slot_to_id  (n_vectors × u64 LE)      │
└──────────────────────────────────────┘
```

On load, the reverse `id → slot` map is rebuilt in memory. Duplicate ids in the `slot_to_id` table are rejected as corrupt.

`n_calib = 0` in the TQ+ trailer means identity calibration (a lazy index with no `add` yet, or a pre-TQ+ index that was re-saved); otherwise it equals `dim`. Loading a version-2 file (no TQ+ trailer) is still supported and is read as identity calibration; version 1 (headerless, no magic) is rejected.

`dim = 0` in the core header signals a lazy uncommitted index. It is only valid alongside `n_vectors = 0`; on load it produces an index whose `dim` is `None` until the first `add` / `add_with_ids` call.

Both formats carry a magic + version byte and are stable across minor versions. Breaking changes bump the version byte.
