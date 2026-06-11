<p align="center">
  <img src="https://raw.githubusercontent.com/a-tokyo/turbovec/main/docs/header.png" alt="turbovec — Google's TurboQuant for vector search" width="100%">
</p>

<p align="center">
  <a href="https://github.com/a-tokyo/turbovec/blob/main/turbovec-node/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
  <a href="https://www.npmjs.com/package/turbovec"><img src="https://img.shields.io/npm/v/turbovec?label=npm&color=blue" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/turbovec"><img src="https://img.shields.io/npm/dt/turbovec?color=blue" alt="npm downloads"></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/paper-arXiv-b31b1b.svg" alt="TurboQuant paper"></a>
</p>

---

**Fast vector quantization with 2–4 bit compression and SIMD search — a native Node.js addon.**

turbovec is a Rust vector index built on Google Research's [**TurboQuant**](https://arxiv.org/abs/2504.19874) algorithm — a data-oblivious quantizer that matches the Shannon lower bound on distortion, with no codebook training and no separate train phase. This package exposes it to Node.js as a [napi-rs](https://napi.rs/) native addon, with prebuilt binaries for common platforms.

- **Online ingest.** Add vectors, they're indexed — no train step, no parameter tuning, no rebuilds as the corpus grows.
- **Faster than FAISS.** Hand-written NEON (ARM) and AVX-512BW (x86) kernels beat FAISS IndexPQFastScan by 12–20% on ARM and match-or-beat it on x86.
- **Filter at search time.** Pass an id allowlist (or a slot bitmask) to `search()` and the kernel honours it directly — up to `k` results from the allowed set, no over-fetch.
- **Pure local.** No managed service, no data leaving your machine or VPC.

## Install

```bash
npm install turbovec
```

Prebuilt binaries ship for linux x64/arm64 (gnu), macOS arm64, and Windows x64; the right one is selected automatically via `optionalDependencies`. The Linux binaries are self-contained (BLAS is statically linked) and have no system library prerequisites.

> **x86-64 CPU baseline.** The x64 binaries are compiled for the `x86-64-v3` micro-architecture (AVX2, Haswell 2013+). Any CPU that can run the AVX2 fallback kernel runs the whole package; the AVX-512 kernel is gated at runtime and only activates on hardware that supports it.

### Supported platforms

Prebuilt native binaries are published for exactly four targets:

| Platform                    | Target triple               |
| --------------------------- | --------------------------- |
| Linux x64 (glibc)           | `x86_64-unknown-linux-gnu`  |
| Linux arm64 (glibc)         | `aarch64-unknown-linux-gnu` |
| macOS arm64 (Apple Silicon) | `aarch64-apple-darwin`      |
| Windows x64                 | `x86_64-pc-windows-msvc`    |

There is **no source-build fallback**. Unlike the Python package (which publishes an sdist that compiles from source), this Node addon has no compile-on-install path: on any platform without a prebuilt binary — Intel macOS (`x86_64-apple-darwin`), Windows arm64 (`win32-arm64`), or musl-based Linux such as Alpine — install resolves no matching `optionalDependencies` package and the loader throws a clear "native binding not found" error at `require`/`import` time. Use one of the four targets above (e.g. a glibc-based image rather than Alpine in containers).

## Quickstart — `TurboQuantIndex`

Positional index: each vector is identified by its insertion slot (`0..n`). Vectors are passed as a **flat row-major `Float32Array`** of length `n * dim`.

```js
const { TurboQuantIndex } = require('turbovec');

// dim must be a positive multiple of 8. bitWidth ∈ {2, 3, 4} (default 4).
const index = new TurboQuantIndex(8, 4);

// Two 8-dim vectors, flat row-major.
const vectors = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
index.add(vectors);

const query = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
const { scores, indices, nq, k } = index.search(query, 2);
// scores: Float32Array (nq*k), indices: BigInt64Array (nq*k, slot positions).
// Row i of the result is scores.slice(i*k, (i+1)*k).
console.log(nq, k, Array.from(indices)); // 1 2 [ 0n, 1n ]

index.write('index.tv');
const loaded = TurboQuantIndex.load('index.tv');
```

`dim` is optional on the constructor — omit it for a lazy index that commits its dim on the first `add`, in which case you must pass `dim` to that first call:

```js
const lazy = new TurboQuantIndex(); // dim inferred
lazy.add(vectors, 8); // dim required here — throws err.code === 'DIM_REQUIRED' otherwise
```

## Quickstart — `IdMapIndex`

Stable external `u64` ids that survive deletes. Ids are JS `bigint`s, passed as a `BigUint64Array`.

```js
const { IdMapIndex } = require('turbovec');

const index = new IdMapIndex(8, 4);

const vectors = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
index.addWithIds(vectors, BigUint64Array.from([1001n, 1002n]));

const query = new Float32Array([1, 0, 0, 0, 0, 0, 0, 0]);
const { scores, ids, nq, k } = index.search(query, 2);
// ids: BigUint64Array — your external u64 ids (as bigint).
console.log(Array.from(ids)); // [ 1001n, 1002n ]

index.remove(1002n); // O(1) by id; returns true if present
index.contains(1001n); // true

// Restrict results to a candidate id set (hybrid retrieval / ACL / tenant).
const filtered = index.search(query, 2, { allowlist: BigUint64Array.from([1001n]) });

index.write('index.tvim');
const loaded = IdMapIndex.load('index.tvim');
```

## Error handling

Errors thrown by the native layer carry a stable `err.code` string so you can branch programmatically:

```js
try {
  index.add(new Float32Array(7)); // not a multiple of dim
} catch (err) {
  if (err.code === 'VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM') {
    // ...handle bad batch shape
  }
}
```

Note: the constructor throws `DIM_NOT_POSITIVE_MULTIPLE_OF_8` (e.g. `new TurboQuantIndex(7)`), while a lazy first-add with a non-multiple-of-8 dim throws `DIM_NOT_MULTIPLE_OF_8` — catch the right code per path.

See [`docs/api.md`](https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md) for the full error-code table and the JS contracts (flat-buffer layout, result shapes, lazy-dim rules).

## Framework integrations

Optional drop-in vector stores for the JS RAG frameworks. Their framework cores are **optional peer dependencies** — install the one you use:

- **LangChain.js** — `npm install turbovec @langchain/core`, then `import { TurbovecVectorStore } from 'turbovec/langchain'`. See [the LangChain.js docs](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/langchain_js.md).
- **LlamaIndex.TS** — `npm install turbovec @llamaindex/core`, then `import { TurbovecVectorStore } from 'turbovec/llamaindex'`. See [the LlamaIndex.TS docs](https://github.com/RyanCodrai/turbovec/blob/main/docs/integrations/llamaindex_ts.md).

## Links

- [Project README](https://github.com/RyanCodrai/turbovec#readme) — benchmarks, recall, how it works.
- [API reference](https://github.com/RyanCodrai/turbovec/blob/main/docs/api.md) — Python / Rust / JS side by side.
- [TurboQuant paper](https://arxiv.org/abs/2504.19874) (ICLR 2026).

## License

MIT
