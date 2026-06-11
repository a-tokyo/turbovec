/**
 * Tests for the LlamaIndex.TS vector-store integration — JS twins of
 * turbovec-python/tests/test_llama_index.py.
 *
 * Embeds node text up front with a deterministic, text-seeded, near-orthogonal
 * embedder (the same FNV-1a + xorshiftHash32 + Box-Muller scheme as the shared
 * `HashEmbeddings` helper, inlined here so node construction stays synchronous),
 * so correctness assertions (self-match ranks first) hold under 4-bit
 * quantization.
 */
import { afterEach, describe, expect, it } from 'vitest';
import * as os from 'node:os';
import * as path from 'node:path';
import * as fs from 'node:fs';
import { NodeRelationship, ObjectType, TextNode } from '@llamaindex/core/schema';
import {
  FilterCondition,
  FilterOperator,
  VectorStoreQueryMode,
} from '@llamaindex/core/vector-store';
import type { VectorStoreQuery } from '@llamaindex/core/vector-store';

import { IdMapIndex } from '../index.js';
import { TurbovecVectorStore, TurbovecQueryModeUnsupportedError } from '../ts/llamaindex.js';

const DIM = 64;

/** Build a TextNode with a precomputed embedding from its text. */
function makeNode(
  text: string,
  opts: { id?: string; metadata?: Record<string, unknown>; refDocId?: string } = {},
): TextNode {
  const node = new TextNode({
    id_: opts.id,
    text,
    metadata: opts.metadata ?? {},
  });
  node.embedding = hashEmbed(text, DIM);
  if (opts.refDocId !== undefined) {
    node.relationships[NodeRelationship.SOURCE] = {
      nodeId: opts.refDocId,
      nodeType: ObjectType.DOCUMENT,
      metadata: {},
    };
  }
  return node;
}

// Deterministic embedder (FNV-1a + xorshiftHash32 + Box-Muller + L2-normalise),
// matching the shared `HashEmbeddings` helper but synchronous.
// NOTE: must stay byte-stable: changing this function invalidates all seeded test vectors.
function hashEmbed(text: string, dim: number): number[] {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  let s = h >>> 0;
  // xorshiftHash32 inline: must stay byte-stable
  const rng = () => {
    s = (Math.imul(s ^ (s >>> 15), s | 1) ^ 1) >>> 0;
    s ^= s << 3;
    s = ((s ^ (s >>> 12)) * 0x45d9f3b) >>> 0;
    return (s >>> 0) / 0x100000000;
  };
  const v = new Array<number>(dim);
  for (let j = 0; j < dim; j++) {
    const u1 = Math.max(rng(), 1e-10);
    const u2 = rng();
    v[j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  let norm = 0;
  // `j` is bounded by `dim`, the array length, so every access is in-bounds.
  for (let j = 0; j < dim; j++) norm += v[j]! * v[j]!;
  norm = Math.sqrt(norm) + 1e-9;
  for (let j = 0; j < dim; j++) v[j]! /= norm;
  return v;
}

function defaultQuery(
  text: string,
  k: number,
  extra: Partial<VectorStoreQuery> = {},
): VectorStoreQuery {
  return {
    queryEmbedding: hashEmbed(text, DIM),
    similarityTopK: k,
    mode: VectorStoreQueryMode.DEFAULT,
    ...extra,
  };
}

const tmpDirs: string[] = [];
function tmpDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'turbovec-li-'));
  tmpDirs.push(dir);
  return dir;
}
afterEach(() => {
  for (const dir of tmpDirs.splice(0)) fs.rmSync(dir, { recursive: true, force: true });
});

// ── Construction ──────────────────────────────────────────────────────────

describe('construction', () => {
  it('no-arg constructor is lazy (dim null)', () => {
    const store = new TurbovecVectorStore();
    expect(store.client().dim).toBe(null);
    expect(store.storesText).toBe(true);
    expect(store.isEmbeddingQuery).toBe(true);
  });

  it('fromParams with dim is eager', () => {
    const store = TurbovecVectorStore.fromParams(DIM, 4);
    expect(store.client().dim).toBe(DIM);
    expect(store.client().bitWidth).toBe(4);
  });

  it('fromParams without dim is lazy', () => {
    const store = TurbovecVectorStore.fromParams(undefined, 2);
    expect(store.client().dim).toBe(null);
    expect(store.client().bitWidth).toBe(2);
  });

  it('lazy dim locks on first add', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('hello')]);
    expect(store.client().dim).toBe(DIM);
  });
});

// ── Add + query ─────────────────────────────────────────────────────────────

describe('add and query', () => {
  it('add returns node ids; query DEFAULT returns nodes', async () => {
    const store = new TurbovecVectorStore();
    const nodes = ['apple', 'banana', 'cherry'].map((t) => makeNode(t));
    const ids = await store.add(nodes);
    expect(ids).toEqual(nodes.map((n) => n.id_));

    const res = await store.query(defaultQuery('apple', 3));
    expect(res.ids.length).toBe(3);
    expect(res.nodes?.length).toBe(3);
    expect(res.similarities.length).toBe(3);
  });

  it('self-match ranks first (correct neighbour, not just monotonic)', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['alpha', 'beta', 'gamma'].map((t) => makeNode(t)));
    const res = await store.query(defaultQuery('alpha', 3));
    const topNode = res.nodes![0] as TextNode;
    expect(topNode.text).toBe('alpha');
    for (let i = 1; i < res.similarities.length; i++) {
      expect(res.similarities[i - 1]!).toBeGreaterThanOrEqual(res.similarities[i]!);
    }
  });

  it('metadata and text round-trip through query', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('hello', { metadata: { source: 'a', n: 7 } })]);
    const res = await store.query(defaultQuery('hello', 1));
    const node = res.nodes![0] as TextNode;
    expect(node.text).toBe('hello');
    expect(node.metadata.source).toBe('a');
    expect(node.metadata.n).toBe(7);
  });

  it('refDocId preserved through query', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('chunk', { refDocId: 'doc-1' })]);
    const res = await store.query(defaultQuery('chunk', 1));
    expect(res.nodes![0]!.sourceNode?.nodeId).toBe('doc-1');
  });

  it('empty store query returns empty', async () => {
    const store = new TurbovecVectorStore();
    const res = await store.query(defaultQuery('anything', 5));
    expect(res).toEqual({ nodes: [], similarities: [], ids: [] });
  });

  it('k larger than ntotal is clamped', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['one', 'two'].map((t) => makeNode(t)));
    const res = await store.query(defaultQuery('one', 100));
    expect(res.ids.length).toBe(2);
  });

  it('query without embedding raises', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('x')]);
    await expect(
      store.query({ similarityTopK: 1, mode: VectorStoreQueryMode.DEFAULT }),
    ).rejects.toThrow(/queryEmbedding/);
  });

  it('empty queryEmbedding raises (masks upstream embedder bug)', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('x')]);
    await expect(
      store.query({ queryEmbedding: [], similarityTopK: 2, mode: VectorStoreQueryMode.DEFAULT }),
    ).rejects.toThrow(/non-empty pre-computed queryEmbedding/);
  });

  it('mismatched dim against eager index raises', async () => {
    const store = new TurbovecVectorStore({ index: new IdMapIndex(32, 4) });
    await expect(store.add([makeNode('hi')])).rejects.toThrow(/embedding dim/);
  });

  it('returned node always has undefined embedding', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('hello')]);
    const res = await store.query(defaultQuery('hello', 1));
    expect(res.nodes![0]!.embedding).toBeUndefined();
  });
});

// ── Upsert / intra-batch dedup ──────────────────────────────────────────────

describe('upsert and dedup', () => {
  it('upsert replaces same node id (last add wins)', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('v1', { id: 'x', metadata: { tag: 'old' } })]);
    await store.add([makeNode('v2', { id: 'x', metadata: { tag: 'new' } })]);
    expect(store.client().length).toBe(1);
    const [node] = store.getNodes(['x']) as TextNode[];
    expect(node!.text).toBe('v2');
    expect(node!.metadata.tag).toBe('new');
  });

  it('intra-batch duplicate node id raises', async () => {
    const store = new TurbovecVectorStore();
    await expect(
      store.add([makeNode('a', { id: 'dup' }), makeNode('b', { id: 'dup' })]),
    ).rejects.toThrow(/duplicate node_id/);
  });

  it('failed upsert preserves existing node (#89)', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('hello', { id: 'my-id' })]);
    // Second add with a wrong-dim embedding fails validation.
    const bad = new TextNode({ id_: 'my-id', text: 'world' });
    bad.embedding = hashEmbed('world', 32);
    await expect(store.add([bad])).rejects.toThrow();
    const [node] = store.getNodes(['my-id']) as TextNode[];
    expect(node!.text).toBe('hello');
    const res = await store.query(defaultQuery('hello', 5));
    expect(res.ids).toEqual(['my-id']);
  });
});

// ── Delete ──────────────────────────────────────────────────────────────────

describe('delete', () => {
  it('delete by refDocId removes every matching node', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', refDocId: 'doc-1' }),
      makeNode('b', { id: 'b', refDocId: 'doc-1' }),
      makeNode('c', { id: 'c', refDocId: 'doc-2' }),
    ]);
    await store.delete('doc-1');
    expect(
      store
        .getNodes()
        .map((n) => n.id_)
        .sort(),
    ).toEqual(['c']);
    expect(store.client().length).toBe(1);
  });

  it('delete by missing refDocId is a no-op', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('a', { id: 'a', refDocId: 'doc-1' })]);
    await store.delete('ghost');
    expect(store.client().length).toBe(1);
  });

  it('deleteNodes by node id', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['a', 'b', 'c'].map((t) => makeNode(t, { id: t })));
    await store.deleteNodes(['b']);
    expect(
      store
        .getNodes()
        .map((n) => n.id_)
        .sort(),
    ).toEqual(['a', 'c']);
  });

  it('deleteNodes by filter', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', metadata: { tier: 'archived' } }),
      makeNode('b', { id: 'b', metadata: { tier: 'live' } }),
    ]);
    await store.deleteNodes(undefined, {
      filters: [{ key: 'tier', value: 'archived', operator: FilterOperator.EQ }],
    });
    expect(store.getNodes().map((n) => n.id_)).toEqual(['b']);
  });

  it('deleteNodes with no args is a no-op', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('x', { id: 'x' })]);
    await store.deleteNodes();
    expect(store.client().length).toBe(1);
  });
});

// ── getNodes ─────────────────────────────────────────────────────────────────

describe('getNodes', () => {
  it('by node ids', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['a', 'b', 'c'].map((t) => makeNode(t, { id: t })));
    expect(
      store
        .getNodes(['a', 'c'])
        .map((n) => n.id_)
        .sort(),
    ).toEqual(['a', 'c']);
  });

  it('by filters', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', metadata: { k: 1 } }),
      makeNode('b', { id: 'b', metadata: { k: 2 } }),
    ]);
    const got = store.getNodes(undefined, {
      filters: [{ key: 'k', value: 2, operator: FilterOperator.EQ }],
    });
    expect(got.map((n) => n.id_)).toEqual(['b']);
  });

  it('no args returns all', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['a', 'b'].map((t) => makeNode(t, { id: t })));
    expect(store.getNodes().length).toBe(2);
  });
});

// ── Filters in query ─────────────────────────────────────────────────────────

describe('query filters', () => {
  const fiveNode = async (): Promise<TurbovecVectorStore> => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('alpha', { id: 'alpha', metadata: { tier: 'free', year: 2020 } }),
      makeNode('beta', { id: 'beta', metadata: { tier: 'pro', year: 2024 } }),
      makeNode('gamma', { id: 'gamma', metadata: { tier: 'free', year: 2024 } }),
      makeNode('delta', { id: 'delta', metadata: { tier: 'pro', year: 2025 } }),
      makeNode('epsilon', { id: 'epsilon', metadata: { tier: 'pro', year: 2019 } }),
    ]);
    return store;
  };

  it('EQ filter restricts results', async () => {
    const store = await fiveNode();
    const res = await store.query(
      defaultQuery('alpha', 10, {
        filters: { filters: [{ key: 'tier', value: 'pro', operator: FilterOperator.EQ }] },
      }),
    );
    expect(res.ids.length).toBe(3);
    expect(res.nodes!.every((n) => n.metadata.tier === 'pro')).toBe(true);
  });

  it('IN filter', async () => {
    const store = await fiveNode();
    const res = await store.query(
      defaultQuery('alpha', 10, {
        filters: {
          filters: [{ key: 'tier', value: ['free'], operator: FilterOperator.IN }],
        },
      }),
    );
    expect(res.ids.sort()).toEqual(['alpha', 'gamma']);
  });

  it('AND filter intersects', async () => {
    const store = await fiveNode();
    const res = await store.query(
      defaultQuery('alpha', 10, {
        filters: {
          filters: [
            { key: 'tier', value: 'pro', operator: FilterOperator.EQ },
            { key: 'year', value: 2024, operator: FilterOperator.GTE },
          ],
          condition: FilterCondition.AND,
        },
      }),
    );
    expect(res.ids.sort()).toEqual(['beta', 'delta']);
  });

  it('OR filter unions', async () => {
    const store = await fiveNode();
    const res = await store.query(
      defaultQuery('alpha', 10, {
        filters: {
          filters: [
            { key: 'year', value: 2020, operator: FilterOperator.EQ },
            { key: 'year', value: 2019, operator: FilterOperator.EQ },
          ],
          condition: FilterCondition.OR,
        },
      }),
    );
    expect(res.ids.sort()).toEqual(['alpha', 'epsilon']);
  });

  it('no matches returns empty', async () => {
    const store = await fiveNode();
    const res = await store.query(
      defaultQuery('alpha', 10, {
        filters: { filters: [{ key: 'tier', value: 'nope', operator: FilterOperator.EQ }] },
      }),
    );
    expect(res).toEqual({ nodes: [], similarities: [], ids: [] });
  });

  it('selective filter returns top-k from the matching set', async () => {
    const store = await fiveNode();
    const res = await store.query(
      defaultQuery('alpha', 2, {
        filters: { filters: [{ key: 'tier', value: 'pro', operator: FilterOperator.EQ }] },
      }),
    );
    // 3 pro nodes exist; top-2 requested → exactly 2 returned, all pro.
    expect(res.ids.length).toBe(2);
    expect(res.nodes!.every((n) => n.metadata.tier === 'pro')).toBe(true);
  });

  it('NE treats missing key as no-match', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', metadata: { color: 'red' } }),
      makeNode('b', { id: 'b', metadata: {} }),
    ]);
    const res = await store.query(
      defaultQuery('a', 10, {
        filters: { filters: [{ key: 'color', value: 'blue', operator: FilterOperator.NE }] },
      }),
    );
    // 'a' (red != blue) matches; 'b' (missing key) does NOT.
    expect(res.ids).toEqual(['a']);
  });

  it('TEXT_MATCH is case-sensitive substring', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', metadata: { body: 'Hello World' } }),
      makeNode('b', { id: 'b', metadata: { body: 'hello world' } }),
    ]);
    const res = await store.query(
      defaultQuery('a', 10, {
        filters: {
          filters: [{ key: 'body', value: 'Hello', operator: FilterOperator.TEXT_MATCH }],
        },
      }),
    );
    expect(res.ids).toEqual(['a']);
  });

  it('CONTAINS matches list membership', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', metadata: { tags: ['x', 'y'] } }),
      makeNode('b', { id: 'b', metadata: { tags: ['z'] } }),
    ]);
    const res = await store.query(
      defaultQuery('a', 10, {
        filters: { filters: [{ key: 'tags', value: 'y', operator: FilterOperator.CONTAINS }] },
      }),
    );
    expect(res.ids).toEqual(['a']);
  });

  it('IS_EMPTY treats missing key as match', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', metadata: { note: 'x' } }),
      makeNode('b', { id: 'b', metadata: {} }),
    ]);
    const res = await store.query(
      defaultQuery('a', 10, {
        filters: { filters: [{ key: 'note', operator: FilterOperator.IS_EMPTY }] },
      }),
    );
    expect(res.ids).toEqual(['b']);
  });
});

// ── node_ids via deleteNodes-style not applicable; docIds in query ───────────

describe('query docIds', () => {
  it('docIds restrict to matching ref doc id', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', refDocId: 'doc-1' }),
      makeNode('b', { id: 'b', refDocId: 'doc-2' }),
      makeNode('c', { id: 'c', refDocId: 'doc-1' }),
    ]);
    const res = await store.query(defaultQuery('a', 10, { docIds: ['doc-1'] }));
    expect(res.ids.sort()).toEqual(['a', 'c']);
  });

  it('docIds and filters intersect', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('a', { id: 'a', refDocId: 'doc-1', metadata: { tier: 'pro' } }),
      makeNode('b', { id: 'b', refDocId: 'doc-1', metadata: { tier: 'free' } }),
      makeNode('c', { id: 'c', refDocId: 'doc-2', metadata: { tier: 'pro' } }),
    ]);
    const res = await store.query(
      defaultQuery('a', 10, {
        docIds: ['doc-1'],
        filters: { filters: [{ key: 'tier', value: 'pro', operator: FilterOperator.EQ }] },
      }),
    );
    expect(res.ids).toEqual(['a']);
  });
});

// ── Unsupported query mode ───────────────────────────────────────────────────

describe('unsupported query mode', () => {
  it('throws the typed error with .code for MMR', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('a')]);
    const q = defaultQuery('a', 1, { mode: VectorStoreQueryMode.MMR });
    await expect(store.query(q)).rejects.toBeInstanceOf(TurbovecQueryModeUnsupportedError);
    try {
      await store.query(q);
    } catch (e) {
      expect((e as TurbovecQueryModeUnsupportedError).code).toBe('TURBOVEC_QUERY_MODE_UNSUPPORTED');
    }
  });

  it('throws for HYBRID', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('a')]);
    await expect(
      store.query(defaultQuery('a', 1, { mode: VectorStoreQueryMode.HYBRID })),
    ).rejects.toThrow(/full-precision/);
  });
});

// ── Missing-handle guard ─────────────────────────────────────────────────────

describe('missing-handle guard', () => {
  it('query skips results whose handle is missing from the nodestore', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['alpha', 'beta', 'gamma'].map((t) => makeNode(t, { id: t })));
    const internals = store as unknown as {
      nodes: Map<string, unknown>;
      nodeIdToU64: Map<string, bigint>;
      u64ToNodeId: Map<bigint, string>;
    };
    const handleB = internals.nodeIdToU64.get('beta')!;
    internals.nodes.delete('beta');
    internals.u64ToNodeId.delete(handleB);
    const res = await store.query(defaultQuery('alpha', 3));
    expect(res.ids.sort()).toEqual(['alpha', 'gamma']);
  });
});

// ── End-to-end retrieval round-trip ─────────────────────────────────────────
//
// Exercises the `add → query` path that `VectorStoreIndex.fromVectorStore`
// (umbrella `llamaindex` package) would drive through the `BaseVectorStore`
// interface. We use only `@llamaindex/core` primitives here to avoid pulling
// the heavy umbrella. The assertions cover exactly what fromVectorStore
// verifies at the store boundary: that added nodes are returned in the correct
// order on a semantic query and that the reconstructed node carries its
// original text and metadata.

describe('end-to-end retrieval round-trip', () => {
  it('add then query returns the expected node via the BaseVectorStore interface', async () => {
    const store = new TurbovecVectorStore();
    const nodes = [
      makeNode('quantum computing explained', {
        id: 'qc',
        metadata: { topic: 'quantum' },
        refDocId: 'doc-quantum',
      }),
      makeNode('classical machine learning basics', {
        id: 'ml',
        metadata: { topic: 'ml' },
        refDocId: 'doc-ml',
      }),
      makeNode('neural network architectures', {
        id: 'nn',
        metadata: { topic: 'ml' },
        refDocId: 'doc-ml',
      }),
    ];

    const returnedIds = await store.add(nodes);
    expect(returnedIds).toEqual(['qc', 'ml', 'nn']);

    // Query for the quantum node — its deterministic hash embedding is
    // sufficiently distant from the ML/NN nodes that it should rank first.
    const result = await store.query(defaultQuery('quantum computing explained', 3));

    expect(result.ids.length).toBe(3);
    expect(result.nodes!.length).toBe(3);
    expect(result.similarities.length).toBe(3);

    // Self-match must rank first (same text → same embedding).
    const top = result.nodes![0] as TextNode;
    expect(top.id_).toBe('qc');
    expect(top.text).toBe('quantum computing explained');
    expect(top.metadata.topic).toBe('quantum');
    expect(top.sourceNode?.nodeId).toBe('doc-quantum');

    // Similarities are non-increasing.
    for (let i = 1; i < result.similarities.length; i++) {
      expect(result.similarities[i - 1]!).toBeGreaterThanOrEqual(result.similarities[i]!);
    }
  });

  it('round-trip after persist preserves node text, metadata, and refDocId', async () => {
    const dir = tmpDir();
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('the quick brown fox', {
        id: 'fox',
        metadata: { animal: true },
        refDocId: 'doc-1',
      }),
    ]);
    await store.persist(dir);

    const loaded = TurbovecVectorStore.fromPersistDir(dir);
    const result = await loaded.query(defaultQuery('the quick brown fox', 1));
    expect(result.ids).toEqual(['fox']);
    const node = result.nodes![0] as TextNode;
    expect(node.text).toBe('the quick brown fox');
    expect(node.metadata.animal).toBe(true);
    expect(node.sourceNode?.nodeId).toBe('doc-1');
  });
});

// ── Persistence ──────────────────────────────────────────────────────────────

describe('persist / fromPersistDir', () => {
  it('round-trips index + nodestore', async () => {
    const store = new TurbovecVectorStore();
    await store.add([
      makeNode('one', { id: 'one', metadata: { n: 1 }, refDocId: 'd1' }),
      makeNode('two', { id: 'two', metadata: { n: 2 } }),
      makeNode('three', { id: 'three', metadata: { n: 3 } }),
    ]);
    const dir = tmpDir();
    await store.persist(dir);
    expect(fs.existsSync(path.join(dir, 'nodestore.json'))).toBe(true);
    expect(fs.existsSync(path.join(dir, 'index.tvim'))).toBe(true);

    const loaded = TurbovecVectorStore.fromPersistDir(dir);
    const res = await loaded.query(defaultQuery('one', 3));
    const top = res.nodes![0] as TextNode;
    expect(top.text).toBe('one');
    expect(top.metadata.n).toBe(1);
    expect(top.sourceNode?.nodeId).toBe('d1');
  });

  it('nodestore.json carries schema_version and expected fields', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('x', { id: 'x' })]);
    const dir = tmpDir();
    await store.persist(dir);
    const data = JSON.parse(fs.readFileSync(path.join(dir, 'nodestore.json'), 'utf8'));
    expect(data.schema_version).toBe(2);
    expect(Object.keys(data).sort()).toEqual(
      ['bit_width', 'next_u64', 'node_id_to_u64', 'nodes', 'schema_version'].sort(),
    );
  });

  it('load rejects an unknown schema version', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('x', { id: 'x' })]);
    const dir = tmpDir();
    await store.persist(dir);
    const file = path.join(dir, 'nodestore.json');
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    data.schema_version = 99;
    fs.writeFileSync(file, JSON.stringify(data));
    expect(() => TurbovecVectorStore.fromPersistDir(dir)).toThrow(/schema version/);
  });

  it('empty/lazy store persist/load round-trips dim===null and bitWidth', async () => {
    const store = new TurbovecVectorStore({ bitWidth: 2 });
    const dir = tmpDir();
    await store.persist(dir);
    const loaded = TurbovecVectorStore.fromPersistDir(dir);
    expect(loaded.client().dim).toBe(null);
    expect(loaded.client().bitWidth).toBe(2);
    const empty = await loaded.query(defaultQuery('anything', 1));
    expect(empty.ids).toEqual([]);
    await loaded.add([makeNode('new')]);
    expect(loaded.client().dim).toBe(DIM);
  });

  it('load then add assigns fresh handles without collision', async () => {
    const store = new TurbovecVectorStore();
    await store.add(['a', 'b', 'c'].map((t) => makeNode(t, { id: t })));
    const dir = tmpDir();
    await store.persist(dir);
    const loaded = TurbovecVectorStore.fromPersistDir(dir);
    await loaded.add([makeNode('d', { id: 'd' })]);
    expect(
      loaded
        .getNodes()
        .map((n) => n.id_)
        .sort(),
    ).toEqual(['a', 'b', 'c', 'd']);
    const res = await loaded.query(defaultQuery('a', 4));
    expect(res.ids.length).toBe(4);
  });

  it('v1 narrow schema still loads with minimum fidelity', async () => {
    const store = new TurbovecVectorStore();
    await store.add([makeNode('legacy', { id: 'leg', metadata: { k: 1 }, refDocId: 'd1' })]);
    const dir = tmpDir();
    await store.persist(dir);
    // Rewrite the side-car as a v1 entry (no nodeDict; narrow text/metadata).
    const file = path.join(dir, 'nodestore.json');
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    data.schema_version = 1;
    data.nodes.leg = { text: 'legacy', metadata: { k: 1 }, refDocId: 'd1' };
    fs.writeFileSync(file, JSON.stringify(data));
    const loaded = TurbovecVectorStore.fromPersistDir(dir);
    const [node] = loaded.getNodes(['leg']) as TextNode[];
    expect(node!.text).toBe('legacy');
    expect(node!.metadata.k).toBe(1);
    expect(node!.sourceNode?.nodeId).toBe('d1');
  });
});
