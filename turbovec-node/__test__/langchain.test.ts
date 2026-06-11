/**
 * Tests for the LangChain.js vector-store integration — JS twins of
 * turbovec-python/tests/test_langchain.py.
 *
 * Uses LangChain's own `SyntheticEmbeddings` as a real, deterministic test
 * double (dim 64 — a multiple of 8, required by the index). It is injected as
 * a dependency, never monkey-patched.
 */
import { afterEach, describe, expect, it } from 'vitest';
import * as os from 'node:os';
import * as path from 'node:path';
import * as fs from 'node:fs';
import { Document } from '@langchain/core/documents';
import { SyntheticEmbeddings } from '@langchain/core/utils/testing';

import { IdMapIndex } from '../index.js';
import { HashEmbeddings } from './helpers.js';
import type { FilterType } from '../ts/langchain.js';
import { TurbovecVectorStore, TurbovecMMRUnsupportedError } from '../ts/langchain.js';

const DIM = 64;
const newEmb = (): SyntheticEmbeddings => new SyntheticEmbeddings({ vectorSize: DIM });

const tmpDirs: string[] = [];
function tmpDir(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'turbovec-lc-'));
  tmpDirs.push(dir);
  return dir;
}
afterEach(() => {
  for (const dir of tmpDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

// ── Construction / indexing ───────────────────────────────────────────────

describe('fromTexts / fromDocuments', () => {
  it('infers dim and indexes', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['apple', 'banana', 'cherry', 'date'],
      {},
      emb,
      { bitWidth: 4 },
    );
    const results = await store.similaritySearch('apple', 4);
    expect(results.length).toBe(4);
  });

  it('fromDocuments stores all docs', async () => {
    const emb = newEmb();
    const docs = [
      new Document({ pageContent: 'a', metadata: { n: 1 } }),
      new Document({ pageContent: 'b', metadata: { n: 2 } }),
    ];
    const store = await TurbovecVectorStore.fromDocuments(docs, emb);
    const results = await store.similaritySearch('a', 2);
    expect(results.length).toBe(2);
  });

  it('empty store search returns empty', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts([], {}, emb);
    expect(await store.similaritySearch('anything', 5)).toEqual([]);
  });

  it('k larger than ntotal is clamped', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['one', 'two'], {}, emb);
    const results = await store.similaritySearch('one', 100);
    expect(results.length).toBe(2);
  });
});

// ── Similarity search surfaces ────────────────────────────────────────────

describe('similarity search', () => {
  it('returns Documents carrying their id', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b', 'c'], {}, emb, {
      ids: ['id-a', 'id-b', 'id-c'],
    });
    const results = await store.similaritySearch('a', 3);
    expect(new Set(results.map((d) => d.id))).toEqual(new Set(['id-a', 'id-b', 'id-c']));
  });

  it('similaritySearchWithScore returns [Document, score] tuples', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['one', 'two', 'three'],
      [{ source: 'a' }, { source: 'b' }, { source: 'c' }],
      emb,
    );
    const scored = await store.similaritySearchWithScore('one', 3);
    expect(scored.length).toBe(3);
    for (const [doc, score] of scored) {
      expect(typeof score).toBe('number');
      expect(typeof doc.pageContent).toBe('string');
    }
  });

  it('search by vector returns documents', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b', 'c'], {}, emb, {
      ids: ['id-a', 'id-b', 'id-c'],
    });
    const qvec = await emb.embedQuery('a');
    const results = await store.similaritySearchVectorWithScore(qvec, 3);
    expect(new Set(results.map(([d]) => d.id))).toEqual(new Set(['id-a', 'id-b', 'id-c']));
  });

  it('scores are returned in monotonically non-increasing order', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'gamma'], {}, emb);
    const scored = await store.similaritySearchWithScore('alpha', 3);
    const scores = scored.map(([, s]) => s);
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i - 1]!).toBeGreaterThanOrEqual(scores[i]!);
    }
  });

  it('metadata round-trips through search', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['hello', 'world'],
      [{ source: 'a' }, { source: 'b' }],
      emb,
    );
    const scored = await store.similaritySearchWithScore('hello', 2);
    expect(new Set(scored.map(([d]) => d.metadata.source))).toEqual(new Set(['a', 'b']));
  });

  // Twin of test_similarity_search_with_score_returns_descending_scores_and_self_match.
  // HashEmbeddings is deterministic and text-seeded with near-orthogonal
  // vectors for distinct texts, so querying a stored document's exact text
  // yields the identical (well-separated) vector — the self-match survives
  // quantization and must rank #1. Proves we return the right neighbours, not
  // just monotonic scores.
  it('self-match wins (query equals a stored doc ranks first)', async () => {
    const emb = new HashEmbeddings(DIM);
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'gamma'], {}, emb);
    const scored = await store.similaritySearchWithScore('alpha', 3);
    expect(scored[0]![0].pageContent).toBe('alpha');
    const scores = scored.map(([, s]) => s);
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i - 1]!).toBeGreaterThanOrEqual(scores[i]!);
    }
  });

  // Defensive guard: a search result whose handle has no docstore mapping must
  // be skipped, not crash. Simulate by deleting a doc's entry out-of-band
  // (leaving the vector in the native index) so the next search returns a
  // handle we can no longer resolve.
  it('search skips results whose handle is missing from the docstore', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'gamma'], {}, emb, {
      ids: ['a', 'b', 'c'],
    });
    // Reach into the private maps to orphan one handle's docstore entry,
    // mimicking an out-of-band/native index edit. The vector stays in the
    // index, so search will still surface its handle.
    const internals = store as unknown as {
      docs: Map<string, unknown>;
      strToU64: Map<string, bigint>;
      u64ToStr: Map<bigint, string>;
    };
    const handleB = internals.strToU64.get('b')!;
    internals.docs.delete('b');
    internals.u64ToStr.delete(handleB);
    // No throw; the orphaned doc is simply omitted from the results.
    const results = await store.similaritySearch('alpha', 3);
    expect(results.map((d) => d.id).sort()).toEqual(['a', 'c']);
  });
});

// ── Relevance score normalization ─────────────────────────────────────────

describe('relevance scores', () => {
  // similaritySearchWithScore returns raw cosine scores (the base class passes
  // similaritySearchVectorWithScore output through verbatim). Raw cosines can
  // be in [-1, 1] — we assert they are numbers, not that they are in [0, 1].
  it('similaritySearchWithScore returns raw cosine scores (numbers, not remapped)', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['one', 'two', 'three'], {}, emb);
    const results = await store.similaritySearchWithScore('one', 3);
    expect(results.length).toBe(3);
    for (const [, score] of results) {
      expect(typeof score).toBe('number');
      // Raw cosine: [-1, 1] plus tiny quantization noise — not remapped to [0, 1].
      expect(score).toBeGreaterThanOrEqual(-1.1);
      expect(score).toBeLessThanOrEqual(1.1);
    }
  });

  // _selectRelevanceScoreFn must remap cosine → [0, 1] for the relevance-score
  // path. The installed @langchain/core version does not expose
  // similaritySearchWithRelevanceScores as a public method, so we verify the
  // overridden fn directly.
  it('_selectRelevanceScoreFn maps cosine scores to [0, 1]', () => {
    const emb = newEmb();
    const store = new TurbovecVectorStore(emb);
    const relevanceFn = (
      store as unknown as { _selectRelevanceScoreFn(): (s: number) => number }
    )._selectRelevanceScoreFn();
    // Boundary values
    expect(relevanceFn(1)).toBe(1); // cos=1 → (1+1)/2 = 1
    expect(relevanceFn(-1)).toBe(0); // cos=-1 → (-1+1)/2 = 0
    expect(relevanceFn(0)).toBe(0.5); // cos=0 → (0+1)/2 = 0.5
    // Quantization overshoot is clamped
    expect(relevanceFn(1.05)).toBe(1);
    expect(relevanceFn(-1.05)).toBe(0);
    // All outputs must be in [0, 1]
    for (const cos of [-1, -0.5, 0, 0.5, 1]) {
      const r = relevanceFn(cos);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(1);
    }
  });
});

// ── similaritySearchWithRelevanceScores ───────────────────────────────────

describe('similaritySearchWithRelevanceScores', () => {
  // Regression guard: the method must exist as a first-class member (not
  // inherited from the base class, which dropped it in @langchain/core v1).
  it('method exists on the store instance', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a'], {}, emb);
    expect(typeof store.similaritySearchWithRelevanceScores).toBe('function');
  });

  it('returns results with scores in [0, 1]', async () => {
    const emb = new HashEmbeddings(DIM);
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'gamma'], {}, emb);
    const results = await store.similaritySearchWithRelevanceScores('alpha', 3);
    expect(results.length).toBeGreaterThan(0);
    for (const [doc, score] of results) {
      expect(typeof doc.pageContent).toBe('string');
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  it('ranking order matches similaritySearchWithScore (same docs, same order)', async () => {
    const emb = new HashEmbeddings(DIM);
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'gamma'], {}, emb);
    const withScore = await store.similaritySearchWithScore('alpha', 3);
    const withRelevance = await store.similaritySearchWithRelevanceScores('alpha', 3);
    // Same number of results.
    expect(withRelevance.length).toBe(withScore.length);
    // Same doc order (page content must match positionally).
    for (let i = 0; i < withScore.length; i++) {
      expect(withRelevance[i]![0].pageContent).toBe(withScore[i]![0].pageContent);
    }
    // Relevance scores must be non-increasing (same ordering contract as raw scores).
    const relScores = withRelevance.map(([, s]) => s);
    for (let i = 1; i < relScores.length; i++) {
      expect(relScores[i - 1]!).toBeGreaterThanOrEqual(relScores[i]!);
    }
  });

  it('near-identical self-match produces relevance score close to 1', async () => {
    const emb = new HashEmbeddings(DIM);
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'gamma'], {}, emb);
    const results = await store.similaritySearchWithRelevanceScores('alpha', 3);
    // Top result should be alpha itself with a high relevance score.
    expect(results[0]![0].pageContent).toBe('alpha');
    // self-match: raw cosine ≈ 1 (quantization may nudge it slightly), so relevance ≈ 1.
    expect(results[0]![1]).toBeGreaterThan(0.9);
  });
});

// ── Filters ───────────────────────────────────────────────────────────────

describe('filters', () => {
  it('metadata-record filter restricts results', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['alpha', 'beta', 'gamma', 'delta', 'epsilon'],
      [{ tier: 'free' }, { tier: 'pro' }, { tier: 'free' }, { tier: 'pro' }, { tier: 'pro' }],
      emb,
    );
    const results = await store.similaritySearch('alpha', 10, { tier: 'pro' });
    expect(results.length).toBe(3);
    expect(results.every((d) => d.metadata.tier === 'pro')).toBe(true);
  });

  it('callable filter receives a Document (metadata reachable)', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['a', 'b', 'c', 'd'],
      [{ n: 1 }, { n: 2 }, { n: 3 }, { n: 4 }],
      emb,
    );
    const results = await store.similaritySearch(
      'a',
      10,
      (doc) => ((doc.metadata.n as number) ?? 0) > 2,
    );
    expect(new Set(results.map((d) => d.metadata.n))).toEqual(new Set([3, 4]));
  });

  it('callable filter can read page_content', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['alpha', 'beta', 'alphabet'], {}, emb);
    const results = await store.similaritySearch('alpha', 10, (doc) =>
      doc.pageContent.startsWith('alpha'),
    );
    expect(new Set(results.map((d) => d.pageContent))).toEqual(new Set(['alpha', 'alphabet']));
  });

  it('callable filter can read document id', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b', 'c'], {}, emb, {
      ids: ['keep-1', 'drop', 'keep-2'],
    });
    const results = await store.similaritySearch('a', 10, (doc) =>
      (doc.id ?? '').startsWith('keep'),
    );
    expect(new Set(results.map((d) => d.id))).toEqual(new Set(['keep-1', 'keep-2']));
  });

  it('filter with scores', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['a', 'b', 'c'],
      [{ k: 1 }, { k: 2 }, { k: 1 }],
      emb,
    );
    const results = await store.similaritySearchWithScore('a', 10, { k: 1 });
    expect(results.length).toBe(2);
    for (const [doc, score] of results) {
      expect(doc.metadata.k).toBe(1);
      expect(typeof score).toBe('number');
    }
  });

  it('filter with no matches returns empty', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b'], [{ k: 1 }, { k: 2 }], emb);
    expect(await store.similaritySearch('a', 5, { k: 999 })).toEqual([]);
  });

  // Twin of test_similarity_search_filter_invalid_type_raises: a non
  // record / non callable filter (here a number) is rejected with a TypeError.
  it('invalid filter type rejects with TypeError', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a'], {}, emb);
    await expect(
      // Force an invalid filter past the type system to exercise the runtime guard.
      store.similaritySearch('a', 1, 42 as unknown as FilterType),
    ).rejects.toBeInstanceOf(TypeError);
  });
});

// ── Upsert / dedup hardening ──────────────────────────────────────────────

describe('upsert and dedup', () => {
  it('upsert replaces metadata and content (last add wins)', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts([], {}, emb);
    await store.addDocuments([
      new Document({ id: 'x', pageContent: 'v1', metadata: { tag: 'old' } }),
    ]);
    await store.addDocuments([
      new Document({ id: 'x', pageContent: 'v2', metadata: { tag: 'new' } }),
    ]);
    const [doc] = store.getByIds(['x']);
    expect(doc!.metadata).toEqual({ tag: 'new' });
    expect(doc!.pageContent).toBe('v2');
  });

  it('re-ingesting an unchanged corpus is idempotent', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts([], {}, emb);
    const docs = [
      new Document({ id: 'a', pageContent: 'hello' }),
      new Document({ id: 'b', pageContent: 'world' }),
    ];
    await store.addDocuments(docs);
    await store.addDocuments(docs);
    expect(store.getByIds(['a', 'b']).length).toBe(2);
  });

  it('does not mutate caller Documents', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts([], {}, emb);
    const docs = [
      new Document({ pageContent: 'a', metadata: { k: 1 } }),
      new Document({ id: 'explicit', pageContent: 'b', metadata: { k: 2 } }),
    ];
    const metaRefs = docs.map((d) => d.metadata);
    await store.addDocuments(docs);
    expect(docs[0]!.id).toBeUndefined();
    expect(docs[1]!.id).toBe('explicit');
    expect(docs.map((d) => d.metadata)).toEqual(metaRefs);
  });

  // #90 twin: intra-batch duplicate ids keep the last occurrence; no orphan.
  it('intra-batch duplicate ids keep last (issue #90)', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts([], {}, emb);
    const docs = [
      new Document({ id: 'dup', pageContent: 'alpha' }),
      new Document({ id: 'dup', pageContent: 'beta' }),
    ];
    const ret = await store.addDocuments(docs);
    // Return value mirrors the input (one entry per input document).
    expect(ret).toEqual(['dup', 'dup']);
    // Exactly one document; last occurrence wins.
    const found = store.getByIds(['dup']);
    expect(found.length).toBe(1);
    expect(found[0]!.pageContent).toBe('beta');
    // No orphaned vector: only one result ever comes back.
    const results = await store.similaritySearch('dup', 10);
    expect(results.length).toBe(1);
  });

  // #89 twin: a failed/invalid upsert batch must leave prior data intact.
  it('failed upsert preserves existing data (issue #89)', async () => {
    // An embedder whose dimension changes between calls forces the second
    // (upsert) add to fail validation.
    let call = 0;
    const varDimEmb = {
      async embedDocuments(texts: string[]): Promise<number[][]> {
        call += 1;
        const dim = call === 1 ? 64 : 32;
        return texts.map((_t, i) => Array<number>(dim).fill(i + 1));
      },
      async embedQuery(): Promise<number[]> {
        return Array<number>(64).fill(1);
      },
    };
    const store = new TurbovecVectorStore(varDimEmb);
    await store.addDocuments([new Document({ id: 'my-id', pageContent: 'hello' })]);
    await expect(
      store.addDocuments([new Document({ id: 'my-id', pageContent: 'world' })]),
    ).rejects.toThrow();
    // Original data survives the failed upsert.
    const [doc] = store.getByIds(['my-id']);
    expect(doc!.pageContent).toBe('hello');
    // Still retrievable via search.
    const results = await store.similaritySearch('hello', 5);
    expect(results.length).toBe(1);
    expect(results[0]!.id).toBe('my-id');
  });

  // FIX 5 twin of Python's ValueError on ids-length mismatch: providing an ids
  // array whose length differs from the documents array must throw immediately
  // (not silently UUID-fill the tail). Tests both under- and over-specification.
  it('options.ids shorter than documents throws (off-by-one)', async () => {
    const emb = newEmb();
    const store = new TurbovecVectorStore(emb);
    const docs = [
      new Document({ pageContent: 'a' }),
      new Document({ pageContent: 'b' }),
      new Document({ pageContent: 'c' }),
    ];
    await expect(store.addDocuments(docs, { ids: ['only-one', 'only-two'] })).rejects.toThrow(
      /options\.ids length/,
    );
  });

  it('options.ids longer than documents throws', async () => {
    const emb = newEmb();
    const store = new TurbovecVectorStore(emb);
    const docs = [new Document({ pageContent: 'a' })];
    await expect(store.addDocuments(docs, { ids: ['id-a', 'extra-id'] })).rejects.toThrow(
      /options\.ids length/,
    );
  });

  it('mismatched dim against an eager index raises', async () => {
    const emb = newEmb();
    const store = new TurbovecVectorStore(emb, { index: new IdMapIndex(32, 4) });
    await expect(store.addDocuments([new Document({ pageContent: 'hi' })])).rejects.toThrow(
      /embedding dimension/,
    );
  });
});

// ── get / delete ──────────────────────────────────────────────────────────

describe('get and delete', () => {
  it('getByIds preserves order, empty input returns []', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b', 'c'], {}, emb, {
      ids: ['id-a', 'id-b', 'id-c'],
    });
    expect(store.getByIds([])).toEqual([]);
    const docs = store.getByIds(['id-c', 'id-a', 'id-b']);
    expect(docs.map((d) => d.id)).toEqual(['id-c', 'id-a', 'id-b']);
  });

  it('delete removes documents', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['apple', 'banana', 'cherry'], {}, emb, {
      ids: ['a', 'b', 'c'],
    });
    await expect(store.delete({ ids: ['b'] })).resolves.toBeUndefined();
    expect(store.getByIds(['a', 'b', 'c']).map((d) => d.id)).toEqual(['a', 'c']);
  });

  it('delete missing ids silently skips', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b'], {}, emb, {
      ids: ['id-a', 'id-b'],
    });
    await store.delete({ ids: ['id-a', 'ghost'] });
    expect(store.getByIds(['id-a']).length).toBe(0);
    expect(store.getByIds(['id-b']).length).toBe(1);
  });

  it('delete with no ids is a no-op', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['x'], {}, emb);
    await expect(store.delete()).resolves.toBeUndefined();
    await expect(store.delete({})).resolves.toBeUndefined();
    expect((await store.similaritySearch('x', 5)).length).toBe(1);
  });
});

// ── Persistence ───────────────────────────────────────────────────────────

describe('save / load', () => {
  it('round-trips index + docstore', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['one', 'two', 'three'],
      [{ n: 1 }, { n: 2 }, { n: 3 }],
      emb,
    );
    const dir = tmpDir();
    await store.save(dir);
    expect(fs.existsSync(path.join(dir, 'docstore.json'))).toBe(true);
    expect(fs.existsSync(path.join(dir, 'index.tvim'))).toBe(true);

    const loaded = await TurbovecVectorStore.load(dir, emb);
    const results = await loaded.similaritySearch('one', 3);
    expect(new Set(results.map((d) => d.pageContent))).toEqual(new Set(['one', 'two', 'three']));
  });

  it('docstore.json carries schema_version and is plain JSON', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['x'], {}, emb);
    const dir = tmpDir();
    await store.save(dir);
    const data = JSON.parse(fs.readFileSync(path.join(dir, 'docstore.json'), 'utf8'));
    expect(data.schema_version).toBeGreaterThanOrEqual(1);
    // Field-for-field with the Python writer.
    expect(Object.keys(data).sort()).toEqual(
      ['bit_width', 'docs', 'next_u64', 'schema_version', 'str_to_u64'].sort(),
    );
  });

  it('load rejects an unknown schema version', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['x'], {}, emb);
    const dir = tmpDir();
    await store.save(dir);
    const file = path.join(dir, 'docstore.json');
    const data = JSON.parse(fs.readFileSync(file, 'utf8'));
    data.schema_version = 99;
    fs.writeFileSync(file, JSON.stringify(data));
    await expect(TurbovecVectorStore.load(dir, emb)).rejects.toThrow(/schema version/);
  });

  // Twin of test_dump_and_load_empty_store: a store with no adds is in the
  // lazy-uncommitted state (dim === null). save/load must round-trip that
  // without losing bitWidth or accidentally committing a dim; a later add
  // commits the dim.
  it('empty/lazy store save/load round-trips dim===null and bitWidth', async () => {
    const emb = newEmb();
    const store = new TurbovecVectorStore(emb, { bitWidth: 2 });
    const dir = tmpDir();
    await store.save(dir);
    const loaded = await TurbovecVectorStore.load(dir, emb);
    const internals = loaded as unknown as { index: IdMapIndex };
    expect(internals.index.dim).toBe(null);
    expect(internals.index.bitWidth).toBe(2);
    // Subsequent search returns empty; subsequent add commits the dim.
    expect(await loaded.similaritySearch('anything', 1)).toEqual([]);
    await loaded.addDocuments([new Document({ pageContent: 'new' })]);
    expect(internals.index.dim).toBe(DIM);
  });

  it('load then add assigns fresh handles without collision', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b', 'c'], {}, emb, {
      ids: ['id-a', 'id-b', 'id-c'],
    });
    const dir = tmpDir();
    await store.save(dir);
    const loaded = await TurbovecVectorStore.load(dir, emb);
    await loaded.addDocuments([new Document({ id: 'id-d', pageContent: 'd' })]);
    const docs = loaded.getByIds(['id-a', 'id-b', 'id-c', 'id-d']);
    expect(docs.map((d) => d.id)).toEqual(['id-a', 'id-b', 'id-c', 'id-d']);
    // All four reachable via search (no handle collision corrupted results).
    const results = await loaded.similaritySearch('a', 4);
    expect(results.length).toBe(4);
  });
});

// ── MMR raises typed error ────────────────────────────────────────────────

describe('max marginal relevance', () => {
  it('throws the typed unsupported error', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(['a', 'b'], {}, emb);
    await expect(store.maxMarginalRelevanceSearch()).rejects.toBeInstanceOf(
      TurbovecMMRUnsupportedError,
    );
    await expect(store.maxMarginalRelevanceSearch()).rejects.toThrow(/full-precision/);
    try {
      await store.maxMarginalRelevanceSearch();
    } catch (e) {
      expect((e as TurbovecMMRUnsupportedError).code).toBe('TURBOVEC_MMR_UNSUPPORTED');
    }
  });
});

// ── Retriever wiring (end-to-end) ─────────────────────────────────────────

describe('asRetriever', () => {
  it('invoke returns documents', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['alpha', 'beta', 'gamma', 'delta'],
      [{ tag: 'a' }, { tag: 'b' }, { tag: 'a' }, { tag: 'b' }],
      emb,
    );
    const retriever = store.asRetriever({ k: 2 });
    const docs = await retriever.invoke('alpha');
    expect(docs.length).toBe(2);
    expect(docs.every((d) => d instanceof Document)).toBe(true);
  });

  it('passes filter through search_kwargs', async () => {
    const emb = newEmb();
    const store = await TurbovecVectorStore.fromTexts(
      ['alpha', 'beta', 'gamma'],
      [{ tag: 'keep' }, { tag: 'drop' }, { tag: 'keep' }],
      emb,
    );
    const retriever = store.asRetriever({ k: 5, filter: { tag: 'keep' } });
    const docs = await retriever.invoke('alpha');
    expect(docs.length).toBe(2);
    expect(docs.every((d) => d.metadata.tag === 'keep')).toBe(true);
  });
});
