# LlamaIndex.TS integration

`turbovec/llamaindex`'s `TurbovecVectorStore` is a LlamaIndex.TS [`BaseVectorStore`](https://ts.llamaindex.ai/docs/llamaindex/modules/data/vector_stores) backed by an `IdMapIndex`. It implements the same public surface as `@llamaindex/core`'s in-tree `SimpleVectorStore` and can be used as a drop-in replacement wherever the simple in-memory store is used.

It is built against `@llamaindex/core` (the `BaseVectorStore`, `VectorStoreQuery` and node-serialization helpers live there; the umbrella `llamaindex` package re-exports the same symbols).

## Install

```bash
npm install turbovec @llamaindex/core
```

`@llamaindex/core` is an **optional peer dependency** — importing `turbovec/llamaindex` without it installed throws a clear error.

## Basic usage

turbovec operates on **pre-computed embeddings** (`isEmbeddingQuery = true`): each node carries its own `embedding`, and `query` requires a `queryEmbedding`. The store never embeds text itself — the calling component (retriever / query engine / ingestion pipeline) is responsible for embedding.

```ts
import { TextNode } from '@llamaindex/core/schema';
import { VectorStoreQueryMode } from '@llamaindex/core/vector-store';
import { TurbovecVectorStore } from 'turbovec/llamaindex';

const store = new TurbovecVectorStore();

await store.add([
  new TextNode({ text: 'hello world', embedding: [/* ...dim floats... */] }),
]);

const result = await store.query({
  queryEmbedding: [/* ...dim floats... */],
  similarityTopK: 5,
  mode: VectorStoreQueryMode.DEFAULT,
});
// result.nodes, result.similarities, result.ids
```

The vector dimensionality is inferred from the first `add()` call.

## Construction

```ts
// No-arg: lazy. dim is inferred from the first add.
const store1 = new TurbovecVectorStore();

// fromParams: same lazy behaviour, plus an explicit bitWidth.
const store2 = TurbovecVectorStore.fromParams(undefined, 4);

// Eager: a known dim.
const store3 = TurbovecVectorStore.fromParams(1536, 4);

// Pre-built index: bring your own IdMapIndex (e.g. one you loaded from disk).
import { IdMapIndex } from 'turbovec';
const store4 = new TurbovecVectorStore({ index: new IdMapIndex(1536, 4) });
```

`bitWidth` is `2`, `3`, or `4` and is fixed once the index is created.

## Wiring into an index

`TurbovecVectorStore` is a `BaseVectorStore`, so it plugs into `VectorStoreIndex` / `storageContextFromDefaults` like any other store.

> **Note:** The snippet below requires the `llamaindex` umbrella package (`npm install llamaindex`), which is **not** installed in this repository's test suite. `VectorStoreIndex` and `storageContextFromDefaults` live in the umbrella and are not re-exported by `@llamaindex/core`. The snippet is illustrative — the store's `add()` / `query()` interface (exercised in the test suite) is the same interface `VectorStoreIndex` calls under the hood.

```ts
// Requires: npm install llamaindex
import { Settings, VectorStoreIndex, storageContextFromDefaults } from 'llamaindex';
import { TurbovecVectorStore } from 'turbovec/llamaindex';

// The retriever/index needs an embed model to embed your documents + queries.
Settings.embedModel = /* your BaseEmbedding, e.g. OpenAIEmbedding */;

const vectorStore = new TurbovecVectorStore();
const storageContext = await storageContextFromDefaults({ vectorStore });
const index = await VectorStoreIndex.fromDocuments(documents, { storageContext });
const retriever = index.asRetriever({ similarityTopK: 5 });
```

## Delete

### `delete(refDocId)` — remove an entire source document

Removes **every node** whose source-document id (`refDocId`) matches. Missing ids are silently ignored.

```ts
await store.delete('my-source-document-123');
```

### `deleteNodes(nodeIds?, filters?)` — remove specific chunks

Removes nodes matching `nodeIds`, `filters`, or both (intersected). Missing node ids are silently ignored.

```ts
import { FilterOperator } from '@llamaindex/core/vector-store';

await store.deleteNodes(['abc-123', 'def-456']);
await store.deleteNodes(undefined, {
  filters: [{ key: 'tier', value: 'archived', operator: FilterOperator.EQ }],
});
```

## Get nodes

```ts
const nodes = store.getNodes(['chunk-1', 'chunk-2']);
const filtered = store.getNodes(undefined, {
  filters: [{ key: 'tier', value: 'pro', operator: FilterOperator.EQ }],
});
```

Returns `BaseNode[]` reconstructed from the side-car. Missing node ids are silently skipped.

## Query

### Supported modes

Only `VectorStoreQueryMode.DEFAULT` is supported. Any other mode (`MMR`, `SPARSE`, `HYBRID`, `SVM`, …) throws `TurbovecQueryModeUnsupportedError` (with `.code === 'TURBOVEC_QUERY_MODE_UNSUPPORTED'`):

```ts
import { TurbovecQueryModeUnsupportedError } from 'turbovec/llamaindex';

try {
  await store.query({ queryEmbedding, similarityTopK: 5, mode: VectorStoreQueryMode.MMR });
} catch (e) {
  if (e instanceof TurbovecQueryModeUnsupportedError) {
    // e.code === 'TURBOVEC_QUERY_MODE_UNSUPPORTED'
  }
}
```

MMR / SVM / hybrid modes need access to full-precision vectors (for pairwise diversity, learned scoring, or sparse-dense fusion) which turbovec discards after quantization.

### Filtered query

`VectorStoreQuery` accepts `filters` and `docIds`. Both intersect when supplied:

```ts
import { FilterCondition, FilterOperator } from '@llamaindex/core/vector-store';

const result = await store.query({
  queryEmbedding,
  similarityTopK: 5,
  mode: VectorStoreQueryMode.DEFAULT,
  filters: {
    filters: [
      { key: 'tier', value: 'pro', operator: FilterOperator.EQ },
      { key: 'year', value: 2024, operator: FilterOperator.GTE },
    ],
    condition: FilterCondition.AND,
  },
  docIds: ['src-doc-42'], // restrict to chunks of this source document
});
```

Supported operators: `EQ`, `NE`, `GT`, `LT`, `GTE`, `LTE`, `IN`, `NIN`, `TEXT_MATCH`, `CONTAINS`, `ANY`, `ALL`, `IS_EMPTY`. Conditions: `AND`, `OR`.

Filter semantics match `SimpleVectorStore`'s reference implementation — every operator except `IS_EMPTY` returns `false` when the filter key is missing, and `TEXT_MATCH` is a case-sensitive substring match. Filters resolve to a handle allowlist **before** scoring, so a selective filter still returns up to `similarityTopK` matches from the filtered set.

## Upsert semantics

Calling `add()` with a node whose `id_` already exists **replaces** the existing entry (the new embedding wins). A failed/invalid `add()` batch never destroys prior data — new vectors are added before old ones are removed.

A node id repeated **within a single `add()` batch** throws an `Error` — deduplicate before calling. (This differs from the LangChain store, which silently keeps the last occurrence; here it's a hard error so an accidental duplicate doesn't quietly drop a node.)

## Persist / load

```ts
await store.persist('./store');
// ... later ...
const reloaded = TurbovecVectorStore.fromPersistDir('./store');
```

`persist(directory)` writes two files into the directory:

- `index.tvim` — the binary `IdMapIndex`.
- `nodestore.json` — a plain-JSON side-car (never pickle/eval) holding node text, metadata, the node-id↔handle map, and a `schema_version`.

The side-car schema is `{ schema_version, nodes, node_id_to_u64, next_u64, bit_width }` — conceptually aligned schema (shared `schema_version` number); top-level and per-node field names differ between runtimes. The Node file uses camelCase top-level keys and a LlamaIndex.TS-serialized `nodeDict` per node; the Python file uses snake_case keys (`ref_doc_id`, `node_dict`) and Python's own serialization format. The files are **not cross-loadable** between runtimes. The per-node payload (`nodeDict`) is produced by LlamaIndex.TS's own `nodeToMetadata` and round-trips via `metadataDictToNode`, preserving the full `BaseNode` subtype, relationships, and excluded-metadata keys. v1 side-cars (narrow `{text, metadata, refDocId}`) still load with minimum-fidelity `TextNode` reconstruction.

Node metadata must be JSON-serializable. An empty (lazy, never-added) store round-trips correctly: `dim` stays `null` and `bitWidth` is preserved, and the next `add` commits the dim.

## Compatibility & differences from the Python integration

This integration tracks the **installed `@llamaindex/core` (0.6.23) typed API**, which is narrower than the Python LlamaIndex surface the Python `TurboQuantVectorStore` targets. The gaps below are framework-version differences, not bugs — the JS store implements everything the installed SDK can actually produce.

- **Filter conditions:** `AND` and `OR`. `FilterCondition.NOT` is **not** part of this `@llamaindex/core` version's API and is not supported.
- **Filter operators:** `EQ`, `NE`, `GT`, `LT`, `GTE`, `LTE`, `IN`, `NIN`, `TEXT_MATCH`, `CONTAINS`, `ANY`, `ALL`, `IS_EMPTY`. `FilterOperator.TEXT_MATCH_INSENSITIVE` is **not** exposed by this version's API and is not supported; `TEXT_MATCH` is case-sensitive.
- **Query-level node-id filter:** `VectorStoreQuery` in this version exposes `filters` and `docIds` — there is **no** `nodeIds` field. (Use `getNodes(nodeIds)` / `deleteNodes(nodeIds)` for id-scoped operations off the query path.)
- **Query mode:** only `VectorStoreQueryMode.DEFAULT` is supported; every other mode throws the typed `TurbovecQueryModeUnsupportedError`.
- **No `clear()` / `toDict()` / `fromDict()`:** the JS store does not implement these Python-side helpers. Use `persist()` / `fromPersistDir()` for serialization.
- **`nodestore.json` is not cross-loadable with Python.** The Node and Python side-cars are separate runtime formats: the Node file carries `bit_width` and a LlamaIndex.TS-serialized per-node `nodeDict`, whereas the Python file serializes nodes in its own format. The schema *fields* and version are conceptually aligned, but the files are not interchangeable across runtimes.
- **Handles are JSON numbers.** Internal u64 handles are issued sequentially starting at 1 (`next_u64` holds the last-issued handle) and serialized as plain JSON numbers, which stay within the JS safe-integer range — correct for stores of up to 2^53 nodes.

## Known limitations

- **Only `VectorStoreQueryMode.DEFAULT` is supported.** MMR / SVM / hybrid modes throw `TurbovecQueryModeUnsupportedError` — they need full-precision vectors that turbovec discards after quantization.
- **turbovec does not embed text.** Nodes must carry pre-computed embeddings and `query` requires a `queryEmbedding`. Supply an `embedModel` (or set `Settings.embedModel`) on the index/retriever that drives the store.
- **JSON-serializable metadata only.** Node metadata is stored as JSON in the side-car; non-serializable values fail at persist time — the same constraint as `SimpleVectorStore`.
- **`stores_text = true`.** Node text is kept in the side-car so query results return populated `TextNode`s without a separate docstore.
