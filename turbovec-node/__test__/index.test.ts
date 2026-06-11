/**
 * Tests for TurboQuantIndex — mirrors turbovec-python/tests/test_index.py.
 */
import { describe, it, expect } from 'vitest';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { TurboQuantIndex } from '../index.js';
import { unitVectors, row } from './helpers.js';

// ── Constructor & basic getters ──────────────────────────────────────────

describe('TurboQuantIndex constructor', () => {
  it('reports dim and bit_width', () => {
    const idx = new TurboQuantIndex(128, 4);
    expect(idx.dim).toBe(128);
    expect(idx.bitWidth).toBe(4);
    expect(idx.length).toBe(0);
  });

  it.each([2, 3, 4])('accepts bit_width=%i', (bw) => {
    const idx = new TurboQuantIndex(128, bw);
    expect(idx.bitWidth).toBe(bw);
    const vecs = unitVectors(20, 128);
    idx.add(vecs);
    expect(idx.length).toBe(20);
  });

  it('rejects bad bit_width', () => {
    for (const bw of [0, 1, 5, 8]) {
      expect(() => new TurboQuantIndex(128, bw)).toThrow();
      try {
        new TurboQuantIndex(128, bw);
      } catch (e: any) {
        expect(e.code).toBe('BIT_WIDTH_OUT_OF_RANGE');
      }
    }
  });

  it('rejects dim not multiple of 8', () => {
    for (const d of [0, 1, 4, 7, 9]) {
      expect(() => new TurboQuantIndex(d, 4)).toThrow();
      try {
        new TurboQuantIndex(d, 4);
      } catch (e: any) {
        expect(e.code).toBe('DIM_NOT_POSITIVE_MULTIPLE_OF_8');
      }
    }
  });

  it('lazy constructor: dim=null', () => {
    const idx = new TurboQuantIndex();
    expect(idx.dim).toBeNull();
    expect(idx.length).toBe(0);
  });
});

// ── add ───────────────────────────────────────────────────────────────────

describe('TurboQuantIndex.add', () => {
  it('updates length', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(50, 128));
    expect(idx.length).toBe(50);
  });

  it('is incremental', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(20, 128, 1));
    idx.add(unitVectors(30, 128, 2));
    expect(idx.length).toBe(50);
  });

  it('throws DIM_MISMATCH when explicit dim arg conflicts with index dim', () => {
    const idx = new TurboQuantIndex(128, 4);
    // Pass dim=256 explicitly, conflicting with the committed dim=128.
    expect(() => idx.add(unitVectors(1, 256), 256)).toThrow();
    try {
      idx.add(unitVectors(1, 256), 256);
    } catch (e: any) {
      expect(e.code).toBe('DIM_MISMATCH');
    }
  });

  it('throws VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM for bad buffer', () => {
    const idx = new TurboQuantIndex(128, 4);
    // Buffer of length 5 is not divisible by 128
    expect(() => idx.add(new Float32Array(5))).toThrow();
    try {
      idx.add(new Float32Array(5));
    } catch (e: any) {
      expect(e.code).toBe('VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM');
    }
  });

  it('throws INVALID_INPUT_VALUE for NaN', () => {
    const idx = new TurboQuantIndex(64, 4);
    const data = unitVectors(1, 64).slice();
    data[5] = NaN;
    expect(() => idx.add(data)).toThrow();
    try {
      idx.add(data);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('throws INVALID_INPUT_VALUE for huge magnitude', () => {
    const idx = new TurboQuantIndex(64, 4);
    const data = unitVectors(1, 64).slice();
    data[5] = 1e20;
    expect(() => idx.add(data)).toThrow();
    try {
      idx.add(data);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('lazy index throws DIM_REQUIRED without dim arg', () => {
    const idx = new TurboQuantIndex();
    expect(() => idx.add(unitVectors(1, 128))).toThrow();
    try {
      idx.add(unitVectors(1, 128));
    } catch (e: any) {
      expect(e.code).toBe('DIM_REQUIRED');
    }
  });

  it('lazy index commits dim when dim arg is supplied', () => {
    const idx = new TurboQuantIndex();
    idx.add(unitVectors(5, 128), 128);
    expect(idx.dim).toBe(128);
    expect(idx.length).toBe(5);
  });

  it('throws DIM_NOT_MULTIPLE_OF_8 for non-multiple-of-8 dim on lazy', () => {
    const idx = new TurboQuantIndex();
    expect(() => idx.add(new Float32Array(9), 9)).toThrow();
    try {
      idx.add(new Float32Array(9), 9);
    } catch (e: any) {
      expect(e.code).toBe('DIM_NOT_MULTIPLE_OF_8');
    }
  });
});

// ── search ────────────────────────────────────────────────────────────────

describe('TurboQuantIndex.search', () => {
  it('result shape nq × k', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(100, 128));
    const res = idx.search(unitVectors(5, 128, 99), 10);
    expect(res.nq).toBe(5);
    expect(res.k).toBe(10);
    expect(res.scores.length).toBe(50);
    expect(res.indices.length).toBe(50);
  });

  it('single query', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(100, 128));
    const res = idx.search(unitVectors(1, 128, 99), 5);
    expect(res.nq).toBe(1);
    expect(res.k).toBe(5);
  });

  it('self-query recall@1 ≥ 95%', () => {
    const dim = 256;
    const n = 100;
    const vecs = unitVectors(n, dim, 42);
    const idx = new TurboQuantIndex(dim, 4);
    idx.add(vecs);

    let hits = 0;
    for (let i = 0; i < 20; i++) {
      const q = vecs.slice(i * dim, (i + 1) * dim);
      const res = idx.search(q, 1);
      if (Number(res.indices[0]) === i) hits++;
    }
    expect(hits).toBeGreaterThanOrEqual(19); // ≥ 95 %
  });

  it('batch vs individual query equivalence', () => {
    const idx = new TurboQuantIndex(256, 4);
    idx.add(unitVectors(50, 256, 0));
    const queries = unitVectors(5, 256, 99);
    const batchRes = idx.search(queries, 3);

    for (let i = 0; i < 5; i++) {
      const q = queries.slice(i * 256, (i + 1) * 256);
      const singleRes = idx.search(q, 3);
      const batchRow = row(batchRes.indices, i, 3);
      const singleRow = singleRes.indices;
      for (let j = 0; j < 3; j++) {
        expect(batchRow[j]).toBe(singleRow[j]);
      }
    }
  });

  it('empty eager index returns k=0', () => {
    const idx = new TurboQuantIndex(128, 4);
    const res = idx.search(unitVectors(1, 128), 3);
    expect(res.k).toBe(0);
    expect(res.scores.length).toBe(0);
    expect(res.indices.length).toBe(0);
  });

  it('throws QUERY_DIM_MISMATCH for wrong dim queries', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(5, 128));
    expect(() => idx.search(unitVectors(1, 64), 1)).toThrow();
    try {
      idx.search(unitVectors(1, 64), 1);
    } catch (e: any) {
      expect(e.code).toBe('QUERY_DIM_MISMATCH');
    }
  });

  it('throws DIM_REQUIRED for non-empty search on lazy uninitialized index', () => {
    const idx = new TurboQuantIndex();
    expect(() => idx.search(unitVectors(1, 128), 1)).toThrow();
    try {
      idx.search(unitVectors(1, 128), 1);
    } catch (e: any) {
      expect(e.code).toBe('DIM_REQUIRED');
    }
  });

  it('empty queries returns nq=0 with correct effective_k', () => {
    const idx = new TurboQuantIndex(64, 4);
    idx.add(unitVectors(3, 64));
    const res = idx.search(new Float32Array(0), 5);
    expect(res.nq).toBe(0);
    expect(res.k).toBe(3); // min(k=5, n=3)
  });
});

// ── swapRemove ────────────────────────────────────────────────────────────

describe('TurboQuantIndex.swapRemove', () => {
  it('shrinks length, returns moved index', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(10, 128));
    const moved = idx.swapRemove(3);
    expect(moved).toBe(9);
    expect(idx.length).toBe(9);
  });

  it('last-element swap is a no-op', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(5, 128));
    expect(idx.swapRemove(4)).toBe(4);
    expect(idx.length).toBe(4);
  });

  it('post-remove search works (cache invalidation)', () => {
    const dim = 256;
    const idx = new TurboQuantIndex(dim, 4);
    const vecs = unitVectors(20, dim, 0);
    idx.add(vecs);

    // Prime cache.
    const pre = idx.search(vecs.slice(5 * dim, 6 * dim), 1);
    expect(Number(pre.indices[0])).toBe(5);

    // Delete slot 5 — last vector (index 19) moves into slot 5.
    idx.swapRemove(5);
    expect(idx.length).toBe(19);

    const post = idx.search(vecs.slice(19 * dim, 20 * dim), 1);
    expect(Number(post.indices[0])).toBe(5);
  });

  it('throws INDEX_OUT_OF_RANGE for out-of-bounds idx', () => {
    const idx = new TurboQuantIndex(128, 4);
    idx.add(unitVectors(3, 128));
    expect(() => idx.swapRemove(99)).toThrow();
    try {
      idx.swapRemove(99);
    } catch (e: any) {
      expect(e.code).toBe('INDEX_OUT_OF_RANGE');
    }
  });
});

// ── prepare / write / load ────────────────────────────────────────────────

describe('TurboQuantIndex.prepare + write + load', () => {
  it('prepare is idempotent', () => {
    const idx = new TurboQuantIndex(64, 4);
    idx.add(unitVectors(20, 64));
    idx.prepare();
    idx.prepare();
    expect(idx.length).toBe(20);
  });

  it('write/load round-trip', () => {
    const dim = 128;
    const vecs = unitVectors(80, dim, 7);
    const idx = new TurboQuantIndex(dim, 4);
    idx.add(vecs);
    idx.prepare();

    const tmpPath = path.join(os.tmpdir(), `turbovec-test-${Date.now()}.tv`);
    try {
      idx.write(tmpPath);
      const loaded = TurboQuantIndex.load(tmpPath);

      expect(loaded.length).toBe(80);
      expect(loaded.dim).toBe(dim);
      expect(loaded.bitWidth).toBe(4);

      // Search results match.
      const q = unitVectors(3, dim, 8);
      const origRes = idx.search(q, 10);
      const loadRes = loaded.search(q, 10);
      expect(loadRes.k).toBe(origRes.k);
      for (let i = 0; i < origRes.indices.length; i++) {
        expect(loadRes.indices[i]).toBe(origRes.indices[i]);
      }
    } finally {
      if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    }
  });

  it('load nonexistent file throws IO_ERROR', () => {
    expect(() => TurboQuantIndex.load('/nonexistent/path/does-not-exist.tv')).toThrow();
    try {
      TurboQuantIndex.load('/nonexistent/path/does-not-exist.tv');
    } catch (e: any) {
      expect(e.code).toBe('IO_ERROR');
    }
  });

  // Regression: a malformed .tv (dim not a multiple of 8) used to trip an
  // internal assert in the core, panicking and ABORTING the whole Node
  // process (SIGABRT) across the napi FFI boundary. The core now validates
  // the header at the read layer and returns an io::Error, mapped to
  // IO_ERROR here. The test running to completion proves the process did
  // not abort.
  it('load malformed .tv (dim not multiple of 8) throws IO_ERROR — process survives', () => {
    const parts: Buffer[] = [];
    parts.push(Buffer.from('TVPI'));
    parts.push(Buffer.from([3])); // version
    parts.push(Buffer.from([4])); // bit_width
    const d = Buffer.alloc(4);
    d.writeUInt32LE(12);
    parts.push(d); // dim=12
    const n = Buffer.alloc(4);
    n.writeUInt32LE(2);
    parts.push(n); // n_vectors=2
    parts.push(Buffer.alloc(8)); // (dim/8)*bw*n = 1*4*2 = 8 packed bytes
    parts.push(Buffer.alloc(8)); // 2 f32 scales
    parts.push(Buffer.alloc(4)); // n_calib = 0
    const tmpPath = path.join(os.tmpdir(), `turbovec-malformed-${Date.now()}.tv`);
    fs.writeFileSync(tmpPath, Buffer.concat(parts));
    try {
      expect(() => TurboQuantIndex.load(tmpPath)).toThrow();
      try {
        TurboQuantIndex.load(tmpPath);
      } catch (e: any) {
        expect(e.code).toBe('IO_ERROR');
      }
    } finally {
      fs.unlinkSync(tmpPath);
    }
  });
});

// ── query finiteness pre-validation (regression for SIGABRT bug) ──────────

describe('TurboQuantIndex.search invalid query values', () => {
  function makeIndex(dim: number): TurboQuantIndex {
    const idx = new TurboQuantIndex(dim, 4);
    idx.add(unitVectors(8, dim, 1));
    return idx;
  }

  it('NaN query throws INVALID_INPUT_VALUE — process survives', () => {
    const idx = makeIndex(8);
    const q = new Float32Array([NaN, 1, 1, 1, 1, 1, 1, 1]);
    expect(() => idx.search(q, 1)).toThrow();
    try {
      idx.search(q, 1);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
    // Process did not abort — reaching here proves survival.
  });

  it('Infinity query throws INVALID_INPUT_VALUE — process survives', () => {
    const idx = makeIndex(8);
    const q = new Float32Array([Infinity, 1, 1, 1, 1, 1, 1, 1]);
    expect(() => idx.search(q, 1)).toThrow();
    try {
      idx.search(q, 1);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('1e20 query throws INVALID_INPUT_VALUE — process survives', () => {
    const idx = makeIndex(8);
    const q = new Float32Array([1e20, 1, 1, 1, 1, 1, 1, 1]);
    expect(() => idx.search(q, 1)).toThrow();
    try {
      idx.search(q, 1);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('masked search: NaN query throws INVALID_INPUT_VALUE', () => {
    const idx = makeIndex(8);
    const q = new Float32Array([NaN, 1, 1, 1, 1, 1, 1, 1]);
    const mask = new Array(8).fill(true);
    expect(() => idx.search(q, 1, { mask })).toThrow();
    try {
      idx.search(q, 1, { mask });
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });
});
