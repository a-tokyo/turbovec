/**
 * LlamaIndex.TS `VectorStore` backed by turbovec's quantized native index.
 *
 * Install with: `npm install turbovec @llamaindex/core` (or the umbrella
 * `llamaindex`, which re-exports the same `@llamaindex/core` symbols).
 *
 * The public surface mirrors `@llamaindex/core`'s in-tree `SimpleVectorStore`
 * (and the Python `TurboQuantVectorStore`) so this store can be swapped in
 * wherever the simple in-memory store is used. `@llamaindex/core` is an
 * OPTIONAL peer dependency — importing this module without it installed throws
 * a clear error.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { BaseEmbedding } from '@llamaindex/core/embeddings';
import type { BaseNode, Metadata } from '@llamaindex/core/schema';
import { NodeRelationship, ObjectType, TextNode } from '@llamaindex/core/schema';
import type {
  MetadataFilter,
  MetadataFilters,
  VectorStoreBaseParams,
  VectorStoreQuery,
  VectorStoreQueryResult,
} from '@llamaindex/core/vector-store';
import {
  BaseVectorStore,
  FilterCondition,
  FilterOperator,
  metadataDictToNode,
  nodeToMetadata,
  VectorStoreQueryMode,
} from '@llamaindex/core/vector-store';

// The package's own napi loader. Resolves the platform-specific `.node`
// binary. Bundlers must NOT inline this (see tsup `external`).
import { IdMapIndex } from '../index.js';

// Re-export the native error-code union + type guard so they're reachable via
// the `turbovec/llamaindex` subpath without adding a new package entry point.
export { isTurbovecError, type TurbovecErrorCode } from './errors.js';

const INDEX_FILENAME = 'index.tvim';
const STORE_FILENAME = 'nodestore.json';
/**
 * Bump when the `nodestore.json` shape changes; the loader refuses to
 * deserialize unknown versions. The number matches the Python writer's
 * `_NODES_SCHEMA_VERSION` (v2 carries the full `node_dict` for lossless
 * BaseNode round-trip) so the two writers stay conceptually aligned. v1 is
 * the older narrow `{text, metadata, ref_doc_id}` shape; we accept it on load
 * and reconstruct minimum-fidelity TextNodes, matching Python's behaviour.
 */
const NODES_SCHEMA_VERSION = 2;
const NODES_SCHEMA_COMPAT: readonly number[] = [1, 2];

/**
 * Side-car entry persisted per node. `metadata` and `refDocId` are kept at the
 * top level for fast filter / doc-id lookup on every query hit. `nodeDict` is
 * the framework's canonical metadata representation (`nodeToMetadata`), which
 * `metadataDictToNode` reconstructs into a full BaseNode — preserving
 * relationships, excluded-metadata keys and templates. Field names use
 * camelCase here (Node.js convention); the Python writer uses snake_case
 * (`ref_doc_id`, `node_dict`) — the schemas are conceptually aligned but are
 * not cross-loadable between runtimes.
 */
interface NodeStoreEntry {
  metadata: Metadata;
  refDocId: string | null;
  nodeDict?: Metadata;
  /** v1-only narrow fallback fields (present on stores predating nodeDict). */
  text?: string;
}

/**
 * Shape of the on-disk `nodestore.json`. `node_id_to_u64` is a list of
 * `[nodeId, handle]` pairs (JSON object keys must be strings; the list form
 * preserves type fidelity, matching Python). Handles are sequential u64 values
 * issued from 0, so they stay within the JS `Number` safe-integer range
 * (< 2^53) for any realistic store — we serialize them as plain JSON numbers.
 * This is NOT a byte-compatible cross-runtime format (LlamaIndex.TS and Python
 * serialize BaseNodes differently); the schema *fields* and version are shared,
 * the per-node `nodeDict` payload is runtime-specific.
 */
interface NodeStorePayload {
  schema_version: number;
  nodes: Record<string, NodeStoreEntry>;
  node_id_to_u64: Array<[string, number]>;
  next_u64: number;
  bit_width: number;
}

/** Construction options for {@link TurbovecVectorStore}. */
export interface TurbovecVectorStoreArgs extends VectorStoreBaseParams {
  /**
   * Pre-built native index. When omitted a lazy `IdMapIndex` is created that
   * commits to a dimensionality on the first add — matching the no-arg
   * ergonomics of LlamaIndex's other vector stores.
   */
  index?: IdMapIndex;
  /** Quantization width (2, 3, or 4). Ignored when `index` is supplied. */
  bitWidth?: number;
}

/**
 * Error thrown when an unsupported query mode is requested. Carries a stable
 * `.code` so callers can branch on it programmatically.
 */
export class TurbovecQueryModeUnsupportedError extends Error {
  readonly code = 'TURBOVEC_QUERY_MODE_UNSUPPORTED' as const;
  constructor(message: string) {
    super(message);
    this.name = 'TurbovecQueryModeUnsupportedError';
  }
}

/**
 * Placeholder embed model. turbovec operates on PRE-COMPUTED embeddings
 * (`isEmbeddingQuery = true`): nodes carry their own embedding and `query`
 * requires a `queryEmbedding`, so the store never embeds text itself. We
 * install this as the default `embedModel` purely so `BaseVectorStore`'s
 * constructor doesn't reach for the throwing global `Settings.embedModel`
 * getter when the caller supplies no model. Calling it throws — matching the
 * contract that turbovec discards full precision and cannot embed.
 */
class NoEmbeddingModel extends BaseEmbedding {
  constructor() {
    super();
  }

  async getTextEmbedding(_text: string): Promise<number[]> {
    throw new Error(
      'TurbovecVectorStore does not embed text — it stores pre-computed ' +
        'embeddings (isEmbeddingQuery=true). Supply an embedModel in the ' +
        'constructor args (or set Settings.embedModel) if a calling component ' +
        'needs to embed.',
    );
  }
}

const unsupportedModeMsg = (mode: VectorStoreQueryMode): string =>
  `TurbovecVectorStore does not support query mode ${JSON.stringify(mode)}. ` +
  `Only VectorStoreQueryMode.DEFAULT is supported — MMR / SVM / hybrid modes ` +
  `need access to full-precision vectors which turbovec discards after ` +
  `quantization. Maintain a parallel store with full vectors if you need a ` +
  `non-default scoring mode.`;

/**
 * LlamaIndex.TS `VectorStore` backed by a native `IdMapIndex`.
 *
 * Vectors are quantized to 2–4 bits per dimension. A side-car map holds node
 * text and metadata keyed by node id, so query results return populated
 * `TextNode`s without depending on a separate docstore (`storesText = true`).
 * Deletion is O(1) per node via the underlying index.
 */
export class TurbovecVectorStore extends BaseVectorStore {
  storesText = true;
  override isEmbeddingQuery = true;

  private readonly index: IdMapIndex;
  private readonly bitWidth: number;
  /** nodeId → side-car entry (metadata + refDocId + full node dict). */
  private readonly nodes: Map<string, NodeStoreEntry>;
  /** nodeId → u64 handle. */
  private readonly nodeIdToU64: Map<string, bigint>;
  /** u64 handle → nodeId (kept in sync so search results translate back). */
  private readonly u64ToNodeId: Map<bigint, string>;
  /** Monotonic handle issuer. Persisted so a reload never reuses a handle. */
  private nextU64: bigint;

  constructor(args: TurbovecVectorStoreArgs = {}) {
    // Default the embed model so BaseVectorStore's constructor never reaches
    // for the throwing global `Settings.embedModel`; turbovec uses pre-computed
    // embeddings and only needs a real model if a caller embeds through it.
    super({
      ...args,
      embedModel: args.embedModel ?? args.embeddingModel ?? new NoEmbeddingModel(),
    });
    this.bitWidth = args.bitWidth ?? 4;
    // IdMapIndex supports lazy construction: passing no dim commits the dim on
    // the first add. Matches the no-dim ergonomics of LlamaIndex's stores.
    this.index = args.index ?? new IdMapIndex(undefined, this.bitWidth);
    this.nodes = new Map();
    this.nodeIdToU64 = new Map();
    this.u64ToNodeId = new Map();
    this.nextU64 = 0n;
  }

  /** Build a store with a known `dim` (eager) or lazy when `dim` is omitted. */
  static fromParams(dim?: number, bitWidth = 4): TurbovecVectorStore {
    return new TurbovecVectorStore({ index: new IdMapIndex(dim, bitWidth) });
  }

  /** The underlying native index. Mirrors Python's `client` property. */
  client(): IdMapIndex {
    return this.index;
  }

  private issueHandle(): bigint {
    this.nextU64 += 1n;
    return this.nextU64;
  }

  // ---- Write path ---------------------------------------------------

  /**
   * Add pre-embedded nodes. Reads each node's embedding, issues sequential u64
   * handles, maintains the nodeId↔handle map and the side-car nodestore.
   *
   * - Intra-batch duplicate node ids are rejected loudly (matching Python):
   *   letting them through would orphan the earlier handle and return the wrong
   *   payload for that vector. Deduplicate before calling `add`.
   * - Existing data for a colliding node id is only removed AFTER the new
   *   vectors are successfully added, so a failed/invalid batch never destroys
   *   prior data (issue #89). Fresh handles mean old and new coexist until the
   *   delete.
   */
  async add(nodes: BaseNode[]): Promise<string[]> {
    if (nodes.length === 0) return [];

    const seen = new Set<string>();
    for (const node of nodes) {
      if (seen.has(node.id_)) {
        throw new Error(
          `duplicate node_id ${JSON.stringify(node.id_)} appears multiple times ` +
            `in the input batch; deduplicate before calling add()`,
        );
      }
      seen.add(node.id_);
    }

    const rows = nodes.map((node) => node.getEmbedding());
    // `rows` is non-empty here (we returned early on an empty batch), so index
    // 0 exists.
    const firstRow = rows[0];
    if (firstRow === undefined) return [];
    const dim = firstRow.length;
    // Validate before mutating any existing data so we surface a clean error
    // rather than a native panic.
    const existingDim = this.index.dim;
    if (existingDim !== null && dim !== existingDim) {
      throw new Error(`node embedding dim ${dim} does not match index dim ${existingDim}`);
    }
    for (const row of rows) {
      if (row.length !== dim) {
        throw new Error('all node embeddings must have the same dimension');
      }
    }

    const flat = new Float32Array(rows.length * dim);
    // `i` is bounded by `rows.length`, so `rows[i]` is always defined.
    for (let i = 0; i < rows.length; i++) flat.set(rows[i]!, i * dim);

    const handles = rows.map(() => this.issueHandle());
    const handleArray = BigUint64Array.from(handles);

    // (#89) Add first; if encoding rejects the batch this throws before any
    // existing data is touched.
    this.index.addWithIds(flat, handleArray, dim);

    // Upsert: any node id already present in the STORE is removed so the
    // re-added vector wins (in-place update semantics).
    for (const node of nodes) {
      if (this.nodeIdToU64.has(node.id_)) this.removeNodeById(node.id_);
    }

    const ids: string[] = [];
    // `i` is bounded by `nodes.length`; `handles` is the parallel per-node array.
    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]!;
      const handle = handles[i]!;
      const nid = node.id_;
      this.nodeIdToU64.set(nid, handle);
      this.u64ToNodeId.set(handle, nid);
      this.nodes.set(nid, {
        metadata: { ...node.metadata },
        refDocId: node.sourceNode?.nodeId ?? null,
        // Round-trip via the framework's own helper so retrieval reconstructs
        // the full BaseNode subclass (TextNode / IndexNode) with relationships,
        // excluded-metadata keys and templates intact. `removeText=false` keeps
        // the text inline (we have no separate docstore).
        nodeDict: nodeToMetadata(node, false, undefined, false),
      });
      ids.push(nid);
    }
    return ids;
  }

  // ---- Delete -------------------------------------------------------

  /**
   * Delete every node whose `refDocId` (source document id) matches. Mirrors
   * Python's `delete(ref_doc_id)`. Missing ref doc ids are silently ignored.
   */
  async delete(refDocId: string, _deleteOptions?: object): Promise<void> {
    const matching: string[] = [];
    for (const [nid, entry] of this.nodes) {
      if (entry.refDocId === refDocId) matching.push(nid);
    }
    for (const nid of matching) this.removeNodeById(nid);
  }

  /**
   * Delete nodes by id and/or metadata filters (intersected when both are
   * supplied). Missing node ids are silently skipped. Mirrors Python's
   * `delete_nodes`.
   */
  async deleteNodes(nodeIds?: string[], filters?: MetadataFilters): Promise<void> {
    if ((nodeIds === undefined || nodeIds.length === 0) && filters === undefined) return;
    let candidates = [...this.nodes.entries()];
    if (nodeIds !== undefined) {
      const idSet = new Set(nodeIds);
      candidates = candidates.filter(([nid]) => idSet.has(nid));
    }
    if (filters !== undefined) {
      candidates = candidates.filter(([, entry]) => filtersMatch(entry.metadata, filters));
    }
    for (const [nid] of candidates) this.removeNodeById(nid);
  }

  private removeNodeById(nodeId: string): boolean {
    const handle = this.nodeIdToU64.get(nodeId);
    if (handle === undefined) return false;
    this.nodeIdToU64.delete(nodeId);
    this.u64ToNodeId.delete(handle);
    this.nodes.delete(nodeId);
    this.index.remove(handle);
    return true;
  }

  /** Return the nodes matching `nodeIds` and/or `filters` (intersected). */
  getNodes(nodeIds?: string[], filters?: MetadataFilters): BaseNode[] {
    let candidates = [...this.nodes.entries()];
    if (nodeIds !== undefined) {
      const idSet = new Set(nodeIds);
      candidates = candidates.filter(([nid]) => idSet.has(nid));
    }
    if (filters !== undefined) {
      candidates = candidates.filter(([, entry]) => filtersMatch(entry.metadata, filters));
    }
    return candidates.map(([nid, entry]) => reconstructNode(nid, entry));
  }

  // ---- Read path (query) --------------------------------------------

  /**
   * Search by a pre-computed `queryEmbedding`. Only
   * `VectorStoreQueryMode.DEFAULT` is supported; any other mode throws
   * {@link TurbovecQueryModeUnsupportedError}. `query.filters` and
   * `query.docIds` resolve to a native handle allowlist BEFORE scoring, so a
   * selective filter still returns up to `similarityTopK` matches from the
   * filtered set.
   */
  async query(query: VectorStoreQuery, _options?: object): Promise<VectorStoreQueryResult> {
    if (query.mode !== VectorStoreQueryMode.DEFAULT) {
      throw new TurbovecQueryModeUnsupportedError(unsupportedModeMsg(query.mode));
    }
    if (query.queryEmbedding === undefined || query.queryEmbedding.length === 0) {
      throw new Error(
        'TurbovecVectorStore requires a non-empty pre-computed queryEmbedding ' +
          '(isEmbeddingQuery=true). An empty embedding usually signals an ' +
          'upstream embedder failure.',
      );
    }
    if (this.index.length === 0) {
      return { nodes: [], similarities: [], ids: [] };
    }

    const qvec = Float32Array.from(query.queryEmbedding);
    const hasFilters = query.filters !== undefined || (query.docIds?.length ?? 0) > 0;

    let result;
    if (!hasFilters) {
      const k = Math.min(query.similarityTopK, this.index.length);
      result = this.index.search(qvec, k);
    } else {
      const allowedHandles = this.resolveAllowedHandles(query.filters, query.docIds);
      if (allowedHandles.length === 0) {
        return { nodes: [], similarities: [], ids: [] };
      }
      result = this.index.search(qvec, query.similarityTopK, {
        allowlist: BigUint64Array.from(allowedHandles),
      });
    }

    const nodes: BaseNode[] = [];
    const similarities: number[] = [];
    const ids: string[] = [];
    for (let i = 0; i < result.ids.length; i++) {
      // Defensive guard: if the native layer returns a handle we have no
      // mapping for (e.g. a stale handle after an out-of-band index edit), skip
      // it rather than dereferencing `undefined` and hard-crashing. A missing
      // nodestore entry for a known handle is treated the same way.
      // `i` is bounded by `result.ids.length`; `ids` and `scores` are parallel.
      const nid = this.u64ToNodeId.get(result.ids[i]!);
      if (nid === undefined) continue;
      const entry = this.nodes.get(nid);
      if (entry === undefined) continue;
      nodes.push(reconstructNode(nid, entry));
      similarities.push(result.scores[i]!);
      ids.push(nid);
    }
    return { nodes, similarities, ids };
  }

  /**
   * Resolve `query.filters` and `query.docIds` to the list of internal u64
   * handles that satisfy them. Both intersect when supplied. Empty list means
   * no node matches.
   */
  private resolveAllowedHandles(
    filters: MetadataFilters | undefined,
    docIds: string[] | undefined,
  ): bigint[] {
    let candidates = [...this.nodes.entries()];
    if (docIds !== undefined && docIds.length > 0) {
      const docIdSet = new Set(docIds);
      candidates = candidates.filter(
        ([, entry]) => entry.refDocId !== null && docIdSet.has(entry.refDocId),
      );
    }
    if (filters !== undefined) {
      candidates = candidates.filter(([, entry]) => filtersMatch(entry.metadata, filters));
    }
    return candidates.map(([nid]) => this.nodeIdToU64.get(nid)!);
  }

  // ---- Persistence --------------------------------------------------

  /**
   * Persist the quantized index plus the side-car to a directory. Writes
   * `index.tvim` (binary native index) and `nodestore.json` (plain JSON, never
   * pickle/eval). Node metadata must be JSON-serializable.
   */
  async persist(directory: string): Promise<void> {
    mkdirSync(directory, { recursive: true });
    this.index.write(join(directory, INDEX_FILENAME));

    const nodesPayload: Record<string, NodeStoreEntry> = {};
    for (const [nid, entry] of this.nodes) nodesPayload[nid] = entry;
    const nodeIdToU64: Array<[string, number]> = [];
    for (const [nid, handle] of this.nodeIdToU64) nodeIdToU64.push([nid, Number(handle)]);

    const payload: NodeStorePayload = {
      schema_version: NODES_SCHEMA_VERSION,
      nodes: nodesPayload,
      node_id_to_u64: nodeIdToU64,
      next_u64: Number(this.nextU64),
      bit_width: this.index.bitWidth,
    };
    writeFileSync(join(directory, STORE_FILENAME), JSON.stringify(payload));
  }

  /** Reload a store previously written by {@link persist}. */
  static fromPersistDir(
    directory: string,
    args: Omit<TurbovecVectorStoreArgs, 'index' | 'bitWidth'> = {},
  ): TurbovecVectorStore {
    const raw = readFileSync(join(directory, STORE_FILENAME), 'utf8');
    const state = JSON.parse(raw) as Partial<NodeStorePayload>;
    const version = state.schema_version ?? 0;
    if (!NODES_SCHEMA_COMPAT.includes(version)) {
      throw new Error(
        `nodestore.json has schema version ${version}; ` +
          `this turbovec accepts versions ${JSON.stringify(NODES_SCHEMA_COMPAT)}`,
      );
    }
    const index = IdMapIndex.load(join(directory, INDEX_FILENAME));
    const store = new TurbovecVectorStore({ ...args, index });

    for (const [nid, entry] of Object.entries(state.nodes ?? {})) {
      store.nodes.set(nid, entry);
    }
    for (const [nid, handle] of state.node_id_to_u64 ?? []) {
      const h = BigInt(handle);
      store.nodeIdToU64.set(nid, h);
      store.u64ToNodeId.set(h, nid);
    }
    store.nextU64 = BigInt(state.next_u64 ?? 0);
    return store;
  }
}

// ---- Node reconstruction --------------------------------------------

function reconstructNode(nodeId: string, entry: NodeStoreEntry): BaseNode {
  // v2 entries carry `nodeDict` — round-trip via the framework's own helper so
  // we get the full BaseNode subclass back with every field populated.
  if (entry.nodeDict !== undefined) {
    return metadataDictToNode(entry.nodeDict);
  }
  // v1 fallback: stores persisted before the full-node round-trip landed only
  // have {text, metadata, refDocId}. Reconstruct the minimum-fidelity TextNode.
  const node = new TextNode({
    id_: nodeId,
    text: entry.text ?? '',
    metadata: { ...entry.metadata },
  });
  if (entry.refDocId !== null) {
    node.relationships[NodeRelationship.SOURCE] = {
      nodeId: entry.refDocId,
      nodeType: ObjectType.DOCUMENT,
      metadata: {},
    };
  }
  return node;
}

// ---- Metadata filtering ---------------------------------------------
//
// Semantics mirror SimpleVectorStore's reference filter implementation so
// filtered results agree with the in-tree store: every operator except
// IS_EMPTY returns false when the key is absent.

function filtersMatch(metadata: Metadata, filters: MetadataFilters): boolean {
  const condition = filters.condition ?? FilterCondition.AND;
  const results = filters.filters.map((f) => singleFilterMatch(metadata, f));
  if (condition === FilterCondition.AND) {
    return results.length > 0 ? results.every(Boolean) : true;
  }
  if (condition === FilterCondition.OR) {
    return results.length > 0 ? results.some(Boolean) : true;
  }
  throw new Error(
    `filter condition ${JSON.stringify(condition)} not supported by TurbovecVectorStore`,
  );
}

function singleFilterMatch(metadata: Metadata, f: MetadataFilter): boolean {
  const op = f.operator;
  const target = f.value;
  const value = metadata[f.key];

  if (op === FilterOperator.IS_EMPTY) {
    return value === undefined || value === null || value === '' || isEmptyArray(value);
  }
  // Every other operator returns false when the key is absent.
  if (value === undefined || value === null) return false;

  switch (op) {
    case FilterOperator.EQ:
      return value === target;
    case FilterOperator.NE:
      return value !== target;
    case FilterOperator.GT:
      return (value as number) > (target as number);
    case FilterOperator.LT:
      return (value as number) < (target as number);
    case FilterOperator.GTE:
      return (value as number) >= (target as number);
    case FilterOperator.LTE:
      return (value as number) <= (target as number);
    case FilterOperator.IN:
      return Array.isArray(target) && (target as unknown[]).includes(value);
    case FilterOperator.NIN:
      return Array.isArray(target) && !(target as unknown[]).includes(value);
    case FilterOperator.CONTAINS:
      return Array.isArray(value) && (value as unknown[]).includes(target);
    case FilterOperator.TEXT_MATCH:
      if (typeof target === 'string' && typeof value === 'string') return value.includes(target);
      throw new TypeError('Both metadata value and filter value must be strings for TEXT_MATCH');
    case FilterOperator.ANY:
      return (
        Array.isArray(target) &&
        Array.isArray(value) &&
        (target as unknown[]).some((t) => (value as unknown[]).includes(t))
      );
    case FilterOperator.ALL:
      return (
        Array.isArray(target) &&
        Array.isArray(value) &&
        (target as unknown[]).every((t) => (value as unknown[]).includes(t))
      );
    default:
      throw new Error(`filter operator ${JSON.stringify(op)} not supported by TurbovecVectorStore`);
  }
}

function isEmptyArray(value: unknown): boolean {
  return Array.isArray(value) && value.length === 0;
}
