/**
 * Tests for mask= and allowlist= filtering — mirrors
 * turbovec-python/tests/test_filtering.py.
 */
import { describe, it, expect } from 'vitest';
import { TurboQuantIndex, IdMapIndex } from '../index.js';
import { unitVectors, row } from './helpers.js';

const DIM = 128;

// ── TurboQuantIndex mask= ─────────────────────────────────────────────────

describe('TurboQuantIndex mask filtering', () => {
  it('mask omitted matches unmasked', () => {
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(100, DIM, 1));
    const q = unitVectors(3, DIM, 2);

    const r1 = idx.search(q, 10);
    // No options arg = no mask
    const r2 = idx.search(q, 10, {});

    expect(r1.indices).toEqual(r2.indices);
    expect(r1.scores).toEqual(r2.scores);
  });

  it('mask all-true matches unmasked', () => {
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(80, DIM, 3));
    const q = unitVectors(2, DIM, 4);

    const r1 = idx.search(q, 5);
    const mask = new Array(idx.length).fill(true);
    const r2 = idx.search(q, 5, { mask });

    expect(r1.indices).toEqual(r2.indices);
  });

  it('mask restricts returned indices to true slots', () => {
    const n = 200;
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(n, DIM, 5));
    const q = unitVectors(4, DIM, 6);

    const allowed = new Set([3, 7, 19, 42, 88, 121, 150, 175, 198]);
    const mask = Array.from({ length: n }, (_, i) => allowed.has(i));

    const res = idx.search(q, 5, { mask });
    expect(res.k).toBe(5);

    for (let qi = 0; qi < res.nq; qi++) {
      for (let j = 0; j < res.k; j++) {
        const slot = Number(res.indices[qi * res.k + j]);
        expect(allowed.has(slot)).toBe(true);
      }
    }
  });

  it('mask shrinks effective_k to popcount(mask)', () => {
    const n = 100;
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(n, DIM, 7));
    const q = unitVectors(2, DIM, 8);

    const mask = new Array(n).fill(false);
    mask[5] = true;
    mask[10] = true;
    mask[15] = true;

    const res = idx.search(q, 10, { mask });
    expect(res.k).toBe(3); // effective_k = popcount(mask) = 3
  });

  it('all-false mask returns k=0', () => {
    const n = 50;
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(n, DIM, 9));
    const q = unitVectors(2, DIM, 10);
    const mask = new Array(n).fill(false);

    const res = idx.search(q, 5, { mask });
    expect(res.k).toBe(0);
    expect(res.scores.length).toBe(0);
    expect(res.indices.length).toBe(0);
  });

  it('wrong-length mask throws MASK_LENGTH_MISMATCH', () => {
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(50, DIM, 11));
    const q = unitVectors(1, DIM, 12);

    expect(() => idx.search(q, 5, { mask: new Array(10).fill(true) })).toThrow();
    try {
      idx.search(q, 5, { mask: new Array(10).fill(true) });
    } catch (e: any) {
      expect(e.code).toBe('MASK_LENGTH_MISMATCH');
    }
  });

  it('mask result == post-hoc filtering of unmasked search', () => {
    const n = 256;
    const idx = new TurboQuantIndex(DIM, 4);
    idx.add(unitVectors(n, DIM, 15));
    const queries = unitVectors(5, DIM, 16);
    const k = 7;

    // Build a mask: every 3rd slot
    const mask = Array.from({ length: n }, (_, i) => i % 3 === 0);

    // Post-hoc: search without mask then filter
    const unfiltered = idx.search(queries, n);
    const nq = unfiltered.nq;
    const unfilK = unfiltered.k;

    for (let qi = 0; qi < nq; qi++) {
      const unfilRow = Array.from(row(unfiltered.indices, qi, unfilK)).map(Number);
      const unfilScores = Array.from(row(unfiltered.scores, qi, unfilK));
      const kept = unfilRow
        // `j` indexes the parallel `unfilScores` row (same length as `unfilRow`).
        .map((slot, j) => ({ slot, score: unfilScores[j]! }))
        .filter(({ slot }) => mask[slot])
        .slice(0, k);

      // Compare with masked search row
      const maskedRes = idx.search(queries.slice(qi * DIM, (qi + 1) * DIM), k, { mask });
      expect(maskedRes.k).toBe(k);
      // `j` is bounded by `k`; the result rows and `kept` both hold k entries.
      for (let j = 0; j < k; j++) {
        expect(Number(maskedRes.indices[j]!)).toBe(kept[j]!.slot);
        expect(Math.abs(maskedRes.scores[j]! - kept[j]!.score)).toBeLessThan(1e-4);
      }
    }
  });
});

// ── IdMapIndex allowlist= ─────────────────────────────────────────────────

describe('IdMapIndex allowlist filtering', () => {
  it('allowlist omitted matches unfiltered', () => {
    const idx = new IdMapIndex(DIM, 4);
    const ids = BigUint64Array.from(Array.from({ length: 100 }, (_, i) => BigInt(7000 + i)));
    idx.addWithIds(unitVectors(100, DIM, 20), ids);
    const q = unitVectors(2, DIM, 21);

    const r1 = idx.search(q, 10);
    // No options arg = no allowlist
    const r2 = idx.search(q, 10, {});

    expect(r1.ids).toEqual(r2.ids);
    expect(r1.scores).toEqual(r2.scores);
  });

  it('allowlist restricts returned ids', () => {
    const idx = new IdMapIndex(DIM, 4);
    const ids = BigUint64Array.from(Array.from({ length: 100 }, (_, i) => BigInt(1000 + i)));
    idx.addWithIds(unitVectors(100, DIM, 22), ids);
    const q = unitVectors(3, DIM, 23);

    const allowed = BigUint64Array.from([1003n, 1010n, 1042n, 1077n, 1099n]);
    const allowedSet = new Set(Array.from(allowed));

    const res = idx.search(q, 10, { allowlist: allowed });
    expect(res.k).toBe(5); // min(k=10, |allowlist|=5) = 5

    for (let qi = 0; qi < res.nq; qi++) {
      for (let j = 0; j < res.k; j++) {
        // `qi*res.k + j` is bounded by the flat nq×k result buffer.
        const id = res.ids[qi * res.k + j]!;
        expect(allowedSet.has(id)).toBe(true);
      }
    }
  });

  it('empty allowlist throws ALLOWLIST_EMPTY', () => {
    const idx = new IdMapIndex(DIM, 4);
    idx.addWithIds(unitVectors(5, DIM, 24), BigUint64Array.from([1n, 2n, 3n, 4n, 5n]));
    const q = unitVectors(1, DIM, 25);

    expect(() => idx.search(q, 3, { allowlist: new BigUint64Array(0) })).toThrow();
    try {
      idx.search(q, 3, { allowlist: new BigUint64Array(0) });
    } catch (e: any) {
      expect(e.code).toBe('ALLOWLIST_EMPTY');
    }
  });

  it('unknown allowlist id throws ALLOWLIST_UNKNOWN_ID with id preview', () => {
    const idx = new IdMapIndex(DIM, 4);
    idx.addWithIds(unitVectors(5, DIM, 26), BigUint64Array.from([1n, 2n, 3n, 4n, 5n]));
    const q = unitVectors(1, DIM, 27);

    expect(() => idx.search(q, 3, { allowlist: BigUint64Array.from([2n, 999n]) })).toThrow();
    try {
      idx.search(q, 3, { allowlist: BigUint64Array.from([2n, 999n]) });
    } catch (e: any) {
      expect(e.code).toBe('ALLOWLIST_UNKNOWN_ID');
      // Message should mention the unknown id
      expect(e.message).toContain('999');
    }
  });

  it('error message previews up to 5 unknown ids', () => {
    const idx = new IdMapIndex(DIM, 4);
    idx.addWithIds(unitVectors(2, DIM, 0), BigUint64Array.from([1n, 2n]));
    const q = unitVectors(1, DIM, 0);

    const unknown = [100n, 101n, 102n, 103n, 104n, 105n]; // 6 unknown
    try {
      idx.search(q, 1, { allowlist: BigUint64Array.from([1n, ...unknown]) });
    } catch (e: any) {
      expect(e.code).toBe('ALLOWLIST_UNKNOWN_ID');
      // Should contain "..." for overflow
      expect(e.message).toContain('...');
    }
  });
});
