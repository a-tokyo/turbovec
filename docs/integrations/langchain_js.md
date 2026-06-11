# LangChain.js integration

`turbovec/langchain`'s `TurbovecVectorStore` is a [LangChain.js `VectorStore`](https://js.langchain.com/docs/integrations/vectorstores/) backed by a native `IdMapIndex`. It implements the same public surface as `@langchain/core`'s in-memory store and can be used as a drop-in replacement wherever an in-memory store is used. It is the JS twin of the Python `turbovec.langchain.TurboQuantVectorStore`, and the two share a schema-compatible on-disk format (the `docstore.json` has the same `schema_version` and field names; raw bytes differ because Python's `json.dump` and JS's `JSON.stringify` use different separator conventions).

## Install

`@langchain/core` is an **optional peer dependency** — install it alongside turbovec:

```bash
npm install turbovec @langchain/core
```

## Basic usage

```ts
import { OpenAIEmbeddings } from "@langchain/openai";
import { TurbovecVectorStore } from "turbovec/langchain";

const embeddings = new OpenAIEmbeddings();

const store = await TurbovecVectorStore.fromTexts(
  ["Document 1...", "Document 2...", "Document 3..."],
  [{}, {}, {}], // per-text metadata (or a single object applied to all)
  embeddings,
  { bitWidth: 4 },
);

const retriever = store.asRetriever({ k: 5 });
const docs = await retriever.invoke("what is turbovec?");
```

The dimensionality of the underlying quantized index is inferred from the embedding model on the first add — no need to specify it up front. (The dim must be a positive multiple of 8, the index's only constraint.)

## Construction

```ts
import { IdMapIndex } from "turbovec";
import { TurbovecVectorStore } from "turbovec/langchain";

// No-arg: lazy. dim is inferred from the first add.
const store = new TurbovecVectorStore(embeddings);

// fromTexts / fromDocuments: same lazy behaviour, plus immediate ingest.
const store2 = await TurbovecVectorStore.fromTexts(
  texts,
  metadatas,
  embeddings,
  {
    bitWidth: 4,
  },
);

// Pre-built index: bring your own IdMapIndex (e.g. one loaded from disk).
const store3 = new TurbovecVectorStore(embeddings, {
  index: new IdMapIndex(1536, 4),
});
```

`bitWidth` is `2`, `3`, or `4` and is fixed once the index is created.

## Adding with explicit ids

```ts
import { Document } from "@langchain/core/documents";

await store.addDocuments(
  [
    new Document({ id: "doc-a", pageContent: "a", metadata: { source: "x" } }),
    new Document({ id: "doc-b", pageContent: "b", metadata: { source: "y" } }),
  ],
  // ids can also be supplied via options; they take precedence over Document.id.
  { ids: ["doc-a", "doc-b"] },
);

// addDocuments honours per-Document.id, falling back to a UUID per document if
// .id is missing — partial ids are not dropped wholesale, and caller Documents
// are never mutated.
await store.addDocuments([
  new Document({ id: "explicit", pageContent: "..." }),
  new Document({ pageContent: "..." }), // gets a UUID
]);
```

If an id is already present, the add **upserts** — the new vector and metadata replace the old (last write wins). This matches the user expectation that re-indexing a document with the same id should replace it, not duplicate it.

Two hardening guarantees match the Python store:

- **Failed batches never destroy existing data (issue #89).** New vectors are validated and added to the index _before_ the old vectors for colliding ids are removed. If a batch is rejected (e.g. a dimension mismatch), the prior data is left fully intact.
- **Intra-batch duplicate ids keep the last occurrence (issue #90).** Two rows sharing an id in a single call collapse to one entry (last wins), so no vector is orphaned. The returned id array still mirrors the input length.

## Search

```ts
// By string query (embeds the query, then searches)
const docs = await store.similaritySearch("what is turbovec?", 5);

// With scores
const docsAndScores = await store.similaritySearchWithScore("...", 5);

// By raw vector
const qvec = await embeddings.embedQuery("...");
const byVec = await store.similaritySearchVectorWithScore(qvec, 5);
```

`similaritySearchWithScore` returns **raw cosine scores** in `[-1, 1]` (possibly slightly outside that range due to quantization noise) — the LangChain.js convention for `similaritySearchVectorWithScore` is to return raw scores, matching the Python store's `_search_vector`. To get normalized relevance scores in `[0, 1]`, call `similaritySearchWithRelevanceScores` — this is a first-class method on `TurbovecVectorStore` (not inherited from the base class, since `@langchain/core` v1 removed it from `VectorStore`). It embeds the query, calls `similaritySearchVectorWithScore`, and maps each raw score through `_selectRelevanceScoreFn()` (`(sim+1)/2`, clamped to `[0, 1]`).

```ts
const docsWithRelevance = await store.similaritySearchWithRelevanceScores(
  "what is turbovec?",
  5,
);
for (const [doc, relevance] of docsWithRelevance) {
  console.log(relevance, doc.pageContent); // relevance ∈ [0, 1]
}
```

## Document retrieval by id

```ts
const docs = store.getByIds(["doc-a", "doc-c"]);
// Missing ids are silently skipped; results preserve input order.
```

## Delete

```ts
await store.delete({ ids: ["doc-a", "doc-b"] }); // missing ids silently skipped
await store.delete(); // no-op
```

Delete is O(1) per id.

## Filters

`similaritySearch`, `similaritySearchWithScore`, and `similaritySearchVectorWithScore` all accept a `filter`:

```ts
// Record — AND of exact equality on Document.metadata.
const a = await store.similaritySearch("query", 5, {
  source: "manual",
  version: 2,
});

// Callable — predicate over the full Document (id / pageContent / metadata).
const b = await store.similaritySearch(
  "query",
  5,
  (doc) => (doc.metadata.score as number) > 0.8,
);
```

The callable form matches the `(doc: Document) => boolean` convention used by the in-memory store, so predicates ported from there work unchanged. Filters are resolved to an id **allowlist before scoring** — the native kernel only inserts allowed documents into the per-query heap, so you get up to `k` results from the filtered set.

## Save / load

```ts
await store.save("./my-store");
// ... later ...
const reloaded = await TurbovecVectorStore.load("./my-store", embeddings);
```

Writes two files under the given directory:

- `index.tvim` — the binary `IdMapIndex` payload.
- `docstore.json` — JSON-encoded document text, metadata, and id maps.

The `docstore.json` schema **matches the Python writer's format** (same `schema_version` and field names: `schema_version`, `docs`, `str_to_u64`, `next_u64`, `bit_width`), so a store dumped from Python can be loaded in Node and vice-versa. The `str_to_u64` and `next_u64` handle values are written as plain JSON numbers (as Python does); handles are sequential `u64` values issued from 0, so they stay safely within the JS `Number` / `2^53` integer range for any real-world store. The loader refuses to deserialize an unknown `schema_version`. Document metadata must be JSON-serializable.

## Known limitations

- **Max-marginal-relevance search is not supported.** `maxMarginalRelevanceSearch` throws a `TurbovecMMRUnsupportedError` (with `.code === "TURBOVEC_MMR_UNSUPPORTED"`) explaining why. MMR requires the full-precision embedding of each candidate to compute pairwise diversity; turbovec discards full-precision vectors after quantization. If you need MMR, keep a parallel store with the raw embeddings and run MMR over that.
- **Embeddings are not retained.** Search returns `Document` objects with `pageContent` and `metadata`, but the original embedding is not recoverable.
- **JSON-serializable metadata only.** Non-JSON-serializable values fail at save time — the same constraint as the in-tree reference store.
