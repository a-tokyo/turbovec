/**
 * LangChain.js VectorStore backed by turbovec's quantized native index.
 *
 * Install with: `npm install turbovec @langchain/core`.
 *
 * The public surface mirrors `@langchain/core`'s in-tree `MemoryVectorStore`
 * (and the Python `TurboQuantVectorStore`) so this store can be swapped in
 * wherever an in-memory store is used. `@langchain/core` is an OPTIONAL peer
 * dependency — importing this module without it installed throws a clear
 * error.
 */
import { randomUUID } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import type { DocumentInterface } from '@langchain/core/documents';
import { Document } from '@langchain/core/documents';
import type { EmbeddingsInterface } from '@langchain/core/embeddings';
import { VectorStore } from '@langchain/core/vectorstores';

// `@langchain/core` declares this type but does not export it from the
// `vectorstores` entry; redeclare the (stable) shape locally to keep the
// override signatures structurally compatible.
type AddDocumentOptions = Record<string, unknown>;

// The package's own napi loader. Resolves the platform-specific `.node`
// binary. Bundlers must NOT inline this (see tsup `external`).
import { IdMapIndex } from '../index.js';

// Re-export the native error-code union + type guard so they're reachable via
// the `turbovec/langchain` subpath without adding a new package entry point.
export { isTurbovecError, type TurbovecErrorCode } from './errors.js';

const INDEX_FILENAME = 'index.tvim';
const STORE_FILENAME = 'docstore.json';
/**
 * Bump when the `docstore.json` shape changes; the loader refuses to
 * deserialize unknown versions. Matches the Python writer's
 * `_DOCSTORE_SCHEMA_VERSION` so a store dumped from either runtime loads in
 * the other.
 */
const DOCSTORE_SCHEMA_VERSION = 1;

/**
 * Filter for similarity search. Either a metadata allowlist (every key/value
 * must match the document's metadata) or a predicate receiving the full
 * `Document`. Mirrors the Python store's dict-or-callable convention and the
 * in-tree `MemoryVectorStore`.
 */
export type FilterType = ((doc: DocumentInterface) => boolean) | Record<string, unknown>;

/** Side-car entry persisted per document. */
interface DocstoreEntry {
  text: string;
  metadata: Record<string, unknown>;
}

/**
 * Shape of the on-disk `docstore.json`. Field-for-field with Python's writer.
 * `str_to_u64` values and `next_u64` are serialized as plain JSON numbers (as
 * Python does). Handles are sequential u64 values issued from 0, so they stay
 * within the JS `Number` safe-integer range (< 2^53) for any real store; we
 * deliberately keep numbers rather than strings to preserve Python on-disk
 * compatibility.
 */
interface DocstorePayload {
  schema_version: number;
  docs: Record<string, DocstoreEntry>;
  str_to_u64: Record<string, number>;
  next_u64: number;
  bit_width: number;
}

/** Construction options for {@link TurbovecVectorStore}. */
export interface TurbovecVectorStoreArgs {
  /**
   * Pre-built native index. When omitted a lazy `IdMapIndex` is created that
   * commits to a dimensionality on the first add — matching the no-arg
   * ergonomics of the in-memory store.
   */
  index?: IdMapIndex;
  /** Quantization width (2, 3, or 4). Ignored when `index` is supplied. */
  bitWidth?: number;
  /** Rehydration state — used internally by {@link TurbovecVectorStore.load}. */
  docs?: Map<string, DocstoreEntry>;
  strToU64?: Map<string, bigint>;
  nextU64?: bigint;
}

/**
 * Error thrown by the unsupported max-marginal-relevance path. Carries a
 * stable `.code` so callers can branch on it programmatically.
 */
export class TurbovecMMRUnsupportedError extends Error {
  readonly code = 'TURBOVEC_MMR_UNSUPPORTED' as const;
  constructor(message: string) {
    super(message);
    this.name = 'TurbovecMMRUnsupportedError';
  }
}

const MMR_MSG =
  'TurbovecVectorStore does not support max-marginal-relevance search ' +
  'because the underlying quantized index discards full-precision vectors ' +
  'after compression. MMR requires the original embedding for every ' +
  'candidate to compute pairwise diversity. Use similaritySearch / ' +
  'similaritySearchWithScore instead, or maintain a parallel store with ' +
  'full-precision embeddings if you need MMR specifically.';

/**
 * LangChain.js `VectorStore` backed by a native `IdMapIndex`.
 *
 * Vectors are quantized to 2–4 bits per dimension. A side-car map holds the
 * original text and metadata keyed by document id. Deletion is O(1) per id
 * via the underlying index.
 */
export class TurbovecVectorStore extends VectorStore {
  declare FilterType: FilterType;

  declare embeddings: EmbeddingsInterface;

  private readonly index: IdMapIndex;
  private readonly bitWidth: number;
  /** id → (text, metadata). */
  private readonly docs: Map<string, DocstoreEntry>;
  /** string id → u64 handle. */
  private readonly strToU64: Map<string, bigint>;
  /** u64 handle → string id (kept in sync so search results translate back). */
  private readonly u64ToStr: Map<bigint, string>;
  /** Monotonic handle issuer. Persisted so a reload never reuses a handle. */
  private nextU64: bigint;

  constructor(embeddings: EmbeddingsInterface, args: TurbovecVectorStoreArgs = {}) {
    super(embeddings, args);
    this.embeddings = embeddings;
    this.bitWidth = args.bitWidth ?? 4;
    // IdMapIndex supports lazy construction: passing no dim commits the dim on
    // the first add. Matches the in-memory store's no-dim ergonomics.
    this.index = args.index ?? new IdMapIndex(undefined, this.bitWidth);
    this.docs = args.docs ?? new Map<string, DocstoreEntry>();
    this.strToU64 = args.strToU64 ?? new Map<string, bigint>();
    this.u64ToStr = new Map();
    for (const [sid, handle] of this.strToU64) this.u64ToStr.set(handle, sid);
    this.nextU64 = args.nextU64 ?? 0n;
  }

  _vectorstoreType(): string {
    return 'turbovec';
  }

  private issueHandle(): bigint {
    this.nextU64 += 1n;
    return this.nextU64;
  }

  // ---- Relevance score normalization --------------------------------

  /**
   * turbovec returns the raw inner product of unit-normalized vectors —
   * ideally cosine similarity in [-1, 1]. Quantization noise can push that
   * slightly outside the bounds, so clamp after mapping to the [0, 1]
   * relevance scale via `(sim + 1) / 2`. Mirrors the Python store.
   */
  protected _selectRelevanceScoreFn(): (score: number) => number {
    return (sim: number) => Math.max(0, Math.min(1, (sim + 1) / 2));
  }

  // ---- Write path ---------------------------------------------------

  /**
   * Add pre-embedded vectors plus their documents. Issues sequential u64
   * handles, maintains the id↔handle map and the docstore.
   *
   * - Intra-batch duplicate ids keep the LAST occurrence (issue #90), matching
   *   the in-memory store whose dict silently overwrites on a repeated id.
   * - Existing data is only removed AFTER the new vectors are successfully
   *   added, so a failed/invalid upsert batch never destroys prior data
   *   (issue #89).
   */
  async addVectors(
    vectors: number[][],
    documents: DocumentInterface[],
    options?: AddDocumentOptions & { ids?: string[] },
  ): Promise<string[]> {
    if (vectors.length === 0) return [];
    if (vectors.length !== documents.length) {
      throw new Error('vectors and documents must have the same length');
    }

    // Resolve ids: explicit option > Document.id > generated UUID. Per-document
    // fallback so partial ids are honoured (does not mutate caller Documents).
    const optionIds = options?.ids;
    // Mirrors Python's ValueError: ids must cover every document or be omitted
    // entirely. A shorter ids array would silently UUID-fill the tail, masking
    // caller bugs. An exact length check (not just "shorter") catches both
    // under- and over-specification.
    if (optionIds !== undefined && optionIds.length !== documents.length) {
      throw new Error(
        `options.ids length (${optionIds.length}) does not match documents length (${documents.length}); ` +
          `provide exactly one id per document or omit ids entirely`,
      );
    }
    let ids = documents.map((doc, i) => optionIds?.[i] ?? doc.id ?? randomUUID());

    let texts = documents.map((doc) => doc.pageContent);
    let metadatas = documents.map((doc) => ({ ...doc.metadata }));
    let rows = vectors;

    // (#90) Dedup intra-batch duplicate ids, keeping the last occurrence. The
    // returned id list still mirrors the input (one entry per input vector).
    const resultIds = ids;
    if (new Set(ids).size !== ids.length) {
      const lastIndexById = new Map<string, number>();
      ids.forEach((id, i) => lastIndexById.set(id, i));
      const keep = [...lastIndexById.values()].sort((a, b) => a - b);
      // `keep` holds indices drawn from `ids.forEach`, so every `i` is a valid
      // in-bounds index into the parallel input arrays.
      ids = keep.map((i) => ids[i]!);
      texts = keep.map((i) => texts[i]!);
      metadatas = keep.map((i) => metadatas[i]!);
      rows = keep.map((i) => rows[i]!);
    }

    // `rows` is non-empty here (we returned early on an empty batch and dedup
    // always keeps at least one row), so index 0 exists.
    const firstRow = rows[0];
    if (firstRow === undefined) return [];
    const dim = firstRow.length;
    // Validate before mutating any existing data so we surface a clean error
    // rather than a native panic.
    const existingDim = this.index.dim;
    if (existingDim !== null && dim !== existingDim) {
      throw new Error(`embedding dimension ${dim} does not match index dim ${existingDim}`);
    }
    for (const row of rows) {
      if (row.length !== dim) {
        throw new Error('all embedding vectors must have the same dimension');
      }
    }

    const flat = new Float32Array(rows.length * dim);
    // `i` is bounded by `rows.length`, so `rows[i]` is always defined.
    for (let i = 0; i < rows.length; i++) flat.set(rows[i]!, i * dim);

    const handles = rows.map(() => this.issueHandle());
    const handleArray = BigUint64Array.from(handles);

    // (#89) Add first. If encoding rejects the batch (non-finite values, dim
    // mismatch on a lazy index, …) this throws before any existing data is
    // touched. Only after a successful add do we remove old vectors for
    // colliding ids, so a failed upsert never destroys existing data. Fresh
    // handles mean old and new vectors coexist until the delete.
    this.index.addWithIds(flat, handleArray, dim);

    // Upsert: any id that already existed is removed so the re-added vector
    // wins (in-place update semantics).
    const duplicates = ids.filter((id) => this.strToU64.has(id));
    if (duplicates.length > 0) await this.delete({ ids: duplicates });

    // `i` is bounded by `ids.length` and the parallel arrays share that length.
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i]!;
      const handle = handles[i]!;
      this.strToU64.set(id, handle);
      this.u64ToStr.set(handle, id);
      this.docs.set(id, { text: texts[i]!, metadata: metadatas[i]! });
    }
    return resultIds;
  }

  /** Embed `documents` via the configured embeddings, then {@link addVectors}. */
  async addDocuments(
    documents: DocumentInterface[],
    options?: AddDocumentOptions & { ids?: string[] },
  ): Promise<string[]> {
    if (documents.length === 0) return [];
    const vectors = await this.embeddings.embedDocuments(documents.map((doc) => doc.pageContent));
    return this.addVectors(vectors, documents, options);
  }

  // ---- Read path (similarity search) --------------------------------

  /**
   * Search by a pre-computed query vector. Returns `[Document, rawScore]`
   * tuples where `rawScore` is the native cosine similarity in `[-1, 1]`
   * (possibly slightly outside due to quantization noise). LangChain's base
   * class calls this method verbatim from `similaritySearchWithScore`, so
   * returning raw scores here is the convention. Relevance remapping
   * (`(sim+1)/2` → `[0,1]`) is applied by the explicit
   * `similaritySearchWithRelevanceScores` override below (note: @langchain/core
   * v1 dropped the `similaritySearchWithRelevanceScores` path from the base
   * class, so this store provides it as a first-class method and is the only
   * caller of `_selectRelevanceScoreFn()`).
   *
   * Mirrors the Python store's `_search_vector` which returns `float(score)`
   * (raw cosine, no remapping).
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: FilterType,
  ): Promise<[DocumentInterface, number][]> {
    if (this.index.length === 0) return [];
    const qvec = Float32Array.from(query);

    let result;
    if (filter === undefined) {
      const searchK = Math.min(k, this.index.length);
      result = this.index.search(qvec, searchK);
    } else {
      const predicate = this.compileFilter(filter);
      const allowedHandles: bigint[] = [];
      for (const [sid, entry] of this.docs) {
        const doc = new Document({
          id: sid,
          pageContent: entry.text,
          metadata: { ...entry.metadata },
        });
        if (predicate(doc)) allowedHandles.push(this.strToU64.get(sid)!);
      }
      if (allowedHandles.length === 0) return [];
      result = this.index.search(qvec, k, {
        allowlist: BigUint64Array.from(allowedHandles),
      });
    }

    const out: [DocumentInterface, number][] = [];
    for (let i = 0; i < result.ids.length; i++) {
      // Defensive guard: if the native layer ever returns a handle we have no
      // mapping for (e.g. a stale handle after an out-of-band index edit), skip
      // it rather than dereferencing `undefined` and hard-crashing. A missing
      // docstore entry for a known handle is treated the same way.
      // `i` is bounded by `result.ids.length`; `ids` and `scores` are parallel.
      const sid = this.u64ToStr.get(result.ids[i]!);
      if (sid === undefined) continue;
      const entry = this.docs.get(sid);
      if (entry === undefined) continue;
      const doc = new Document({
        id: sid,
        pageContent: entry.text,
        metadata: { ...entry.metadata },
      });
      // Return the raw native score; relevance remapping is applied by the
      // explicit similaritySearchWithRelevanceScores override below.
      out.push([doc, result.scores[i]!]);
    }
    return out;
  }

  /**
   * Search by a string query and return `[Document, relevanceScore]` tuples
   * where `relevanceScore` is remapped to `[0, 1]` via `_selectRelevanceScoreFn()`.
   *
   * This is an explicit first-class override required because `@langchain/core`
   * v1 removed `similaritySearchWithRelevanceScores` from the `VectorStore` base
   * class. `_selectRelevanceScoreFn()` maps raw cosine similarity `sim` to
   * `(sim + 1) / 2`, clamped to `[0, 1]`, mirroring the Python store's
   * relevance convention.
   */
  async similaritySearchWithRelevanceScores(
    query: string,
    k = 4,
    filter?: FilterType,
  ): Promise<[DocumentInterface, number][]> {
    const embedded = await this.embeddings.embedQuery(query);
    const raw = await this.similaritySearchVectorWithScore(embedded, k, filter);
    const relevanceFn = this._selectRelevanceScoreFn();
    return raw.map(([doc, score]) => [doc, relevanceFn(score)]);
  }

  private compileFilter(filter: FilterType): (doc: DocumentInterface) => boolean {
    if (typeof filter === 'function') return filter;
    if (filter !== null && typeof filter === 'object') {
      const entries = Object.entries(filter);
      return (doc) => entries.every(([key, value]) => doc.metadata[key] === value);
    }
    throw new TypeError(
      'filter must be a metadata record or a callable taking a Document, ' + `got ${typeof filter}`,
    );
  }

  // ---- Get / delete -------------------------------------------------

  /** Return Documents for the given ids, in input order. Missing ids skipped. */
  getByIds(ids: string[]): DocumentInterface[] {
    const out: DocumentInterface[] = [];
    for (const sid of ids) {
      const entry = this.docs.get(sid);
      if (entry === undefined) continue;
      out.push(
        new Document({
          id: sid,
          pageContent: entry.text,
          metadata: { ...entry.metadata },
        }),
      );
    }
    return out;
  }

  /** Remove documents by id. Missing ids are silently skipped; `ids` omitted is a no-op. */
  override async delete(params?: { ids?: string[] }): Promise<void> {
    const ids = params?.ids;
    if (ids === undefined || ids.length === 0) return;
    for (const sid of ids) {
      const handle = this.strToU64.get(sid);
      if (handle === undefined) continue;
      this.strToU64.delete(sid);
      this.u64ToStr.delete(handle);
      this.docs.delete(sid);
      this.index.remove(handle);
    }
  }

  // ---- Max marginal relevance ---------------------------------------

  /**
   * Unsupported: the quantized index discards full-precision vectors, which
   * MMR requires for pairwise diversity scoring. Throws
   * {@link TurbovecMMRUnsupportedError}.
   */
  override async maxMarginalRelevanceSearch(): Promise<DocumentInterface[]> {
    throw new TurbovecMMRUnsupportedError(MMR_MSG);
  }

  // ---- Construction helpers -----------------------------------------

  static override async fromTexts(
    texts: string[],
    metadatas: object[] | object,
    embeddings: EmbeddingsInterface,
    dbConfig: TurbovecVectorStoreArgs & { ids?: string[] } = {},
  ): Promise<TurbovecVectorStore> {
    const docs = texts.map((text, i) => {
      const metadata: object = Array.isArray(metadatas)
        ? ((metadatas as object[])[i] ?? {})
        : metadatas;
      return new Document({
        pageContent: text,
        metadata: metadata as Record<string, unknown>,
        ...(dbConfig.ids ? { id: dbConfig.ids[i] } : {}),
      });
    });
    return TurbovecVectorStore.fromDocuments(docs, embeddings, dbConfig);
  }

  static override async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface,
    dbConfig: TurbovecVectorStoreArgs & { ids?: string[] } = {},
  ): Promise<TurbovecVectorStore> {
    const store = new TurbovecVectorStore(embeddings, dbConfig);
    if (docs.length > 0) {
      await store.addDocuments(docs, dbConfig.ids ? { ids: dbConfig.ids } : undefined);
    }
    return store;
  }

  // ---- Persistence --------------------------------------------------

  /**
   * Persist the quantized index plus the side-car to a directory. Writes
   * `index.tvim` (binary native index) and `docstore.json` (plain JSON whose
   * format matches the Python writer — same fields, handle values as JSON
   * numbers safe up to 2^53). Document metadata must be JSON-serializable.
   */
  async save(directory: string): Promise<void> {
    mkdirSync(directory, { recursive: true });
    this.index.write(join(directory, INDEX_FILENAME));

    const docsPayload: Record<string, DocstoreEntry> = {};
    for (const [sid, entry] of this.docs) {
      docsPayload[sid] = { text: entry.text, metadata: entry.metadata };
    }
    const strToU64: Record<string, number> = {};
    for (const [sid, handle] of this.strToU64) strToU64[sid] = Number(handle);

    const payload: DocstorePayload = {
      schema_version: DOCSTORE_SCHEMA_VERSION,
      docs: docsPayload,
      str_to_u64: strToU64,
      next_u64: Number(this.nextU64),
      bit_width: this.index.bitWidth,
    };
    writeFileSync(join(directory, STORE_FILENAME), JSON.stringify(payload));
  }

  /** Reload a store previously written by {@link save}. */
  static async load(
    directory: string,
    embeddings: EmbeddingsInterface,
  ): Promise<TurbovecVectorStore> {
    const raw = readFileSync(join(directory, STORE_FILENAME), 'utf8');
    const state = JSON.parse(raw) as Partial<DocstorePayload>;
    const version = state.schema_version ?? 0;
    if (version !== DOCSTORE_SCHEMA_VERSION) {
      throw new Error(
        `docstore.json has schema version ${version}; ` +
          `this turbovec expects version ${DOCSTORE_SCHEMA_VERSION}`,
      );
    }
    const index = IdMapIndex.load(join(directory, INDEX_FILENAME));

    const docs = new Map<string, DocstoreEntry>();
    for (const [sid, entry] of Object.entries(state.docs ?? {})) {
      docs.set(sid, { text: entry.text, metadata: entry.metadata });
    }
    const strToU64 = new Map<string, bigint>();
    for (const [sid, handle] of Object.entries(state.str_to_u64 ?? {})) {
      strToU64.set(sid, BigInt(handle));
    }
    return new TurbovecVectorStore(embeddings, {
      index,
      bitWidth: state.bit_width ?? 4,
      docs,
      strToU64,
      nextU64: BigInt(state.next_u64 ?? 0),
    });
  }
}
