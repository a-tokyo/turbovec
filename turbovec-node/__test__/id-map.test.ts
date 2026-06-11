/**
 * Tests for IdMapIndex — mirrors turbovec-python/tests/test_id_map.py.
 */
import { describe, it, expect } from 'vitest';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { IdMapIndex, TurboQuantIndex } from '../index.js';
import { unitVectors } from './helpers.js';

// ── Constructor ───────────────────────────────────────────────────────────

describe('IdMapIndex constructor', () => {
  it('rejects bad bit_width', () => {
    for (const bw of [0, 1, 5, 8]) {
      expect(() => new IdMapIndex(128, bw)).toThrow();
      try {
        new IdMapIndex(128, bw);
      } catch (e: any) {
        expect(e.code).toBe('BIT_WIDTH_OUT_OF_RANGE');
      }
    }
  });

  it('rejects dim not multiple of 8', () => {
    for (const d of [0, 1, 4, 7, 9]) {
      expect(() => new IdMapIndex(d, 4)).toThrow();
      try {
        new IdMapIndex(d, 4);
      } catch (e: any) {
        expect(e.code).toBe('DIM_NOT_POSITIVE_MULTIPLE_OF_8');
      }
    }
  });
});

// ── addWithIds ────────────────────────────────────────────────────────────

describe('IdMapIndex.addWithIds', () => {
  it('updates length and contains', () => {
    const idx = new IdMapIndex(128, 4);
    const vecs = unitVectors(5, 128);
    idx.addWithIds(vecs, BigUint64Array.from([10n, 20n, 30n, 40n, 50n]));
    expect(idx.length).toBe(5);
    expect(idx.contains(30n)).toBe(true);
    expect(idx.contains(99n)).toBe(false);
  });

  it('incremental add works', () => {
    const idx = new IdMapIndex(128, 4);
    idx.addWithIds(unitVectors(3, 128, 0), BigUint64Array.from([1n, 2n, 3n]));
    idx.addWithIds(unitVectors(2, 128, 1), BigUint64Array.from([4n, 5n]));
    expect(idx.length).toBe(5);
  });

  it('throws ID_ALREADY_PRESENT for duplicate id', () => {
    const idx = new IdMapIndex(128, 4);
    idx.addWithIds(unitVectors(2, 128), BigUint64Array.from([1n, 2n]));
    expect(() => idx.addWithIds(unitVectors(1, 128, 1), BigUint64Array.from([2n]))).toThrow();
    try {
      idx.addWithIds(unitVectors(1, 128, 1), BigUint64Array.from([2n]));
    } catch (e: any) {
      expect(e.code).toBe('ID_ALREADY_PRESENT');
    }
  });

  it('throws IDS_COUNT_MISMATCH when ids.length != n_vectors', () => {
    const idx = new IdMapIndex(128, 4);
    expect(() => idx.addWithIds(unitVectors(3, 128), BigUint64Array.from([1n, 2n]))).toThrow();
    try {
      idx.addWithIds(unitVectors(3, 128), BigUint64Array.from([1n, 2n]));
    } catch (e: any) {
      expect(e.code).toBe('IDS_COUNT_MISMATCH');
    }
  });

  it('throws INVALID_INPUT_VALUE for NaN vector', () => {
    const idx = new IdMapIndex(64, 4);
    const data = unitVectors(1, 64).slice();
    data[5] = NaN;
    expect(() => idx.addWithIds(data, BigUint64Array.from([1n]))).toThrow();
    try {
      idx.addWithIds(data, BigUint64Array.from([1n]));
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('lazy index throws DIM_REQUIRED without dim arg', () => {
    const idx = new IdMapIndex();
    expect(() => idx.addWithIds(unitVectors(1, 128), BigUint64Array.from([1n]))).toThrow();
    try {
      idx.addWithIds(unitVectors(1, 128), BigUint64Array.from([1n]));
    } catch (e: any) {
      expect(e.code).toBe('DIM_REQUIRED');
    }
  });
});

// ── remove / contains ─────────────────────────────────────────────────────

describe('IdMapIndex.remove', () => {
  it('returns true if present, false otherwise', () => {
    const idx = new IdMapIndex(128, 4);
    idx.addWithIds(unitVectors(3, 128), BigUint64Array.from([1n, 2n, 3n]));
    expect(idx.remove(2n)).toBe(true);
    expect(idx.length).toBe(2);
    expect(idx.remove(2n)).toBe(false); // already gone
    expect(idx.remove(999n)).toBe(false); // never existed
  });

  it('remove then re-add same id', () => {
    const idx = new IdMapIndex(128, 4);
    idx.addWithIds(unitVectors(5, 128), BigUint64Array.from([1n, 2n, 3n, 4n, 5n]));
    expect(idx.remove(3n)).toBe(true);
    idx.addWithIds(unitVectors(1, 128, 42), BigUint64Array.from([3n]));
    expect(idx.contains(3n)).toBe(true);
    expect(idx.length).toBe(5);
  });

  it('remaining ids self-query after removes', () => {
    const dim = 256;
    const idx = new IdMapIndex(dim, 4);
    const vecs = unitVectors(15, dim, 0);
    const ids = BigUint64Array.from(Array.from({ length: 15 }, (_, i) => BigInt(i * 7 + 11)));
    idx.addWithIds(vecs, ids);

    const removedPositions = new Set([5, 14, 0]);
    // Positions are all < 15, the id-array length, so each lookup is in-bounds.
    for (const p of removedPositions) idx.remove(ids[p]!);

    for (let i = 0; i < 15; i++) {
      if (removedPositions.has(i)) continue;
      const q = vecs.slice(i * dim, (i + 1) * dim);
      const res = idx.search(q, 1);
      expect(res.ids[0]).toBe(ids[i]);
    }
  });
});

// ── search ────────────────────────────────────────────────────────────────

describe('IdMapIndex.search', () => {
  it('returns external ids', () => {
    const dim = 256;
    const idx = new IdMapIndex(dim, 4);
    const vecs = unitVectors(10, dim, 0);
    const ids = BigUint64Array.from(Array.from({ length: 10 }, (_, i) => BigInt(1_000_000 + i)));
    idx.addWithIds(vecs, ids);

    const res = idx.search(vecs, 1);
    for (let i = 0; i < 10; i++) {
      expect(res.ids[i]).toBe(ids[i]);
    }
  });

  it('empty eager index returns k=0', () => {
    const idx = new IdMapIndex(128, 4);
    const res = idx.search(unitVectors(1, 128), 3);
    expect(res.k).toBe(0);
    expect(res.ids.length).toBe(0);
  });

  it('throws QUERY_DIM_MISMATCH for wrong-dim queries', () => {
    const idx = new IdMapIndex(128, 4);
    idx.addWithIds(unitVectors(3, 128), BigUint64Array.from([1n, 2n, 3n]));
    expect(() => idx.search(unitVectors(1, 64), 1)).toThrow();
    try {
      idx.search(unitVectors(1, 64), 1);
    } catch (e: any) {
      expect(e.code).toBe('QUERY_DIM_MISMATCH');
    }
  });

  it('empty queries shape contract matches TurboQuantIndex', () => {
    const dim = 64;
    const tq = new TurboQuantIndex(dim, 4);
    const im = new IdMapIndex(dim, 4);
    tq.add(unitVectors(3, dim));
    im.addWithIds(unitVectors(3, dim), BigUint64Array.from([1n, 2n, 3n]));

    const tqRes = tq.search(new Float32Array(0), 5);
    const imRes = im.search(new Float32Array(0), 5);

    expect(tqRes.nq).toBe(0);
    expect(imRes.nq).toBe(0);
    // effective_k = min(k=5, n=3) = 3
    expect(tqRes.k).toBe(3);
    expect(imRes.k).toBe(3);
  });

  it('bigint id > 2^53 round-trips losslessly', () => {
    const dim = 64;
    const idx = new IdMapIndex(dim, 4);
    // Use an id well above Number.MAX_SAFE_INTEGER
    const bigId = 2n ** 55n + 1n;
    const smallId = 1n;
    const vecs = unitVectors(2, dim, 7);
    idx.addWithIds(vecs, BigUint64Array.from([bigId, smallId]));

    const q = vecs.slice(0, dim); // query with the bigId vector
    const res = idx.search(q, 1);
    expect(res.ids[0]).toBe(bigId); // exact bigint equality
  });

  it('empty-queries effective_k dedups allowlist', () => {
    // Mirrors test_search_empty_queries_dedups_allowlist_for_effective_k
    const dim = 64;
    const idx = new IdMapIndex(dim, 4);
    idx.addWithIds(unitVectors(3, dim), BigUint64Array.from([10n, 20n, 30n]));

    // allowlist with 3 copies of the same id — effective n_allowed = 1
    const al = BigUint64Array.from([10n, 10n, 10n]);
    const empty = new Float32Array(0);
    const realQ = unitVectors(1, dim);

    const emptyRes = idx.search(empty, 5, { allowlist: al });
    const realRes = idx.search(realQ, 5, { allowlist: al });

    // Both should have effective_k=1
    expect(emptyRes.k).toBe(realRes.k);
    expect(emptyRes.k).toBe(1);
    expect(realRes.k).toBe(1);
    expect(emptyRes.nq).toBe(0);
    expect(realRes.nq).toBe(1);
  });
});

// ── write / load ──────────────────────────────────────────────────────────

describe('IdMapIndex.write + load', () => {
  it('write/load round-trip with removes', () => {
    const dim = 256;
    const idx = new IdMapIndex(dim, 4);
    const vecs = unitVectors(10, dim, 0);
    const ids = BigUint64Array.from(Array.from({ length: 10 }, (_, i) => BigInt(5000 + i)));
    idx.addWithIds(vecs, ids);
    idx.remove(5004n);
    idx.remove(5007n);

    const tmpPath = path.join(os.tmpdir(), `turbovec-idmap-${Date.now()}.tvim`);
    try {
      idx.write(tmpPath);
      const restored = IdMapIndex.load(tmpPath);

      expect(restored.length).toBe(8);
      expect(restored.contains(5000n)).toBe(true);
      expect(restored.contains(5004n)).toBe(false);
      expect(restored.contains(5007n)).toBe(false);

      // Self-queries for remaining ids.
      for (let i = 0; i < 10; i++) {
        const id = BigInt(5000 + i);
        if (id === 5004n || id === 5007n) continue;
        const q = vecs.slice(i * dim, (i + 1) * dim);
        const res = restored.search(q, 1);
        expect(res.ids[0]).toBe(id);
      }
    } finally {
      if (fs.existsSync(tmpPath)) fs.unlinkSync(tmpPath);
    }
  });

  it('load nonexistent file throws IO_ERROR', () => {
    expect(() => IdMapIndex.load('/nonexistent/path/does-not-exist.tvim')).toThrow();
    try {
      IdMapIndex.load('/nonexistent/path/does-not-exist.tvim');
    } catch (e: any) {
      expect(e.code).toBe('IO_ERROR');
    }
  });

  // Regression: a malformed .tvim (dim not a multiple of 8) used to trip an
  // internal assert in the core, aborting the whole Node process (SIGABRT)
  // across the napi FFI boundary. The core now validates the header and
  // returns an io::Error mapped to IO_ERROR. Test completion proves survival.
  it('load malformed .tvim (dim not multiple of 8) throws IO_ERROR — process survives', () => {
    const parts: Buffer[] = [];
    parts.push(Buffer.from('TVIM'));
    parts.push(Buffer.from([3])); // version
    parts.push(Buffer.from([4])); // bit_width
    const d = Buffer.alloc(4);
    d.writeUInt32LE(12);
    parts.push(d); // dim=12
    const n = Buffer.alloc(4);
    n.writeUInt32LE(2);
    parts.push(n); // n_vectors=2
    parts.push(Buffer.alloc(8)); // (dim/8)*bw*n = 8 packed bytes
    parts.push(Buffer.alloc(8)); // 2 f32 scales
    parts.push(Buffer.alloc(4)); // n_calib = 0
    parts.push(Buffer.alloc(16)); // 2 u64 ids
    const tmpPath = path.join(os.tmpdir(), `turbovec-idmap-malformed-${Date.now()}.tvim`);
    fs.writeFileSync(tmpPath, Buffer.concat(parts));
    try {
      expect(() => IdMapIndex.load(tmpPath)).toThrow();
      try {
        IdMapIndex.load(tmpPath);
      } catch (e: any) {
        expect(e.code).toBe('IO_ERROR');
      }
    } finally {
      fs.unlinkSync(tmpPath);
    }
  });
});

// ── query finiteness pre-validation (regression for SIGABRT bug) ──────────

describe('IdMapIndex.search invalid query values', () => {
  function makeIdIndex(dim: number): IdMapIndex {
    const idx = new IdMapIndex(dim, 4);
    idx.addWithIds(
      unitVectors(8, dim, 1),
      BigUint64Array.from(Array.from({ length: 8 }, (_, i) => BigInt(i + 1))),
    );
    return idx;
  }

  it('NaN query throws INVALID_INPUT_VALUE — process survives', () => {
    const idx = makeIdIndex(8);
    const q = new Float32Array([NaN, 1, 1, 1, 1, 1, 1, 1]);
    expect(() => idx.search(q, 1)).toThrow();
    try {
      idx.search(q, 1);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('Infinity query throws INVALID_INPUT_VALUE — process survives', () => {
    const idx = makeIdIndex(8);
    const q = new Float32Array([Infinity, 1, 1, 1, 1, 1, 1, 1]);
    expect(() => idx.search(q, 1)).toThrow();
    try {
      idx.search(q, 1);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('1e20 query throws INVALID_INPUT_VALUE — process survives', () => {
    const idx = makeIdIndex(8);
    const q = new Float32Array([1e20, 1, 1, 1, 1, 1, 1, 1]);
    expect(() => idx.search(q, 1)).toThrow();
    try {
      idx.search(q, 1);
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });

  it('allowlist search: NaN query throws INVALID_INPUT_VALUE', () => {
    const idx = makeIdIndex(8);
    const q = new Float32Array([NaN, 1, 1, 1, 1, 1, 1, 1]);
    const allowlist = BigUint64Array.from([1n, 2n, 3n]);
    expect(() => idx.search(q, 1, { allowlist })).toThrow();
    try {
      idx.search(q, 1, { allowlist });
    } catch (e: any) {
      expect(e.code).toBe('INVALID_INPUT_VALUE');
    }
  });
});

// ── negative/out-of-range BigInt aliasing regression ─────────────────────

describe('IdMapIndex.remove / contains — negative and oversized BigInt', () => {
  it('remove(-1n) returns false and does NOT delete id 1n', () => {
    const idx = new IdMapIndex(8, 4);
    idx.addWithIds(unitVectors(2, 8, 0), BigUint64Array.from([1n, 2n]));
    expect(idx.remove(-1n)).toBe(false);
    expect(idx.length).toBe(2); // nothing was removed
    expect(idx.contains(1n)).toBe(true);
    expect(idx.contains(2n)).toBe(true);
  });

  it('contains(-1n) returns false while contains(1n) stays true', () => {
    const idx = new IdMapIndex(8, 4);
    idx.addWithIds(unitVectors(1, 8, 0), BigUint64Array.from([1n]));
    expect(idx.contains(-1n)).toBe(false);
    expect(idx.contains(1n)).toBe(true);
  });

  it('contains(-3n) returns false when index holds id 3n', () => {
    const idx = new IdMapIndex(8, 4);
    idx.addWithIds(unitVectors(1, 8, 0), BigUint64Array.from([3n]));
    expect(idx.contains(-3n)).toBe(false);
    expect(idx.contains(3n)).toBe(true);
  });

  it('remove(2n**70n) returns false (truncation guard)', () => {
    const idx = new IdMapIndex(8, 4);
    idx.addWithIds(unitVectors(2, 8, 0), BigUint64Array.from([1n, 2n]));
    expect(idx.remove(2n ** 70n)).toBe(false);
    expect(idx.length).toBe(2);
  });
});
