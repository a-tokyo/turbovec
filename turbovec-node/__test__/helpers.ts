/**
 * Shared test helpers — mirrors the Python `unit_vectors` helper used
 * throughout turbovec-python/tests/*.py.
 */

/**
 * A minimal seeded PRNG so tests are deterministic without external deps.
 * Algorithm: xorshift on (s ^ (s >>> 15)) then multiply-xorshift with
 * constant 0x45d9f3b, producing a uniform float in [0, 1).
 */
export function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s ^ (s >>> 15), s | 1) ^ 1) >>> 0;
    s ^= s << 3;
    s = ((s ^ (s >>> 12)) * 0x45d9f3b) >>> 0;
    return (s >>> 0) / 0x100000000;
  };
}

/**
 * Generate `n` unit vectors of dimension `dim` using a seeded RNG.
 * Returns a flat row-major Float32Array of length n * dim.
 */
export function unitVectors(n: number, dim: number, seed = 0): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(n * dim);

  for (let i = 0; i < n; i++) {
    const row = out.subarray(i * dim, (i + 1) * dim);
    // Fill with Normal(0, 1) via Box-Muller.
    for (let j = 0; j < dim; j++) {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      row[j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    // L2-normalise.
    let norm = 0;
    // `j` is bounded by `dim`, the subarray length, so every access is in-bounds.
    for (let j = 0; j < dim; j++) norm += row[j]! * row[j]!;
    norm = Math.sqrt(norm) + 1e-9;
    for (let j = 0; j < dim; j++) row[j]! /= norm;
  }

  return out;
}

/** Extract row i (0-based) from a flat nq×k Float32Array buffer. */
export function row(buf: Float32Array, i: number, k: number): Float32Array;
/** Extract row i (0-based) from a flat nq×k BigInt64Array buffer. */
export function row(buf: BigInt64Array, i: number, k: number): BigInt64Array;
/** Extract row i (0-based) from a flat nq×k BigUint64Array buffer. */
export function row(buf: BigUint64Array, i: number, k: number): BigUint64Array;
/** Extract row i (0-based) from a flat nq×k typed-array buffer. */
export function row(
  buf: Float32Array | BigInt64Array | BigUint64Array,
  i: number,
  k: number,
): Float32Array | BigInt64Array | BigUint64Array {
  return buf.slice(i * k, (i + 1) * k);
}

/** Extract a row of Float32Array values as a regular JS number array. */
export function rowNumbers(buf: Float32Array, i: number, k: number): number[] {
  return Array.from(row(buf, i, k));
}

/**
 * Deterministic text -> unit-vector embedder for the LangChain tests. JS twin
 * of the Python `StubEmbeddings`: hashes the input string to seed `mulberry32`,
 * then draws a Normal(0,1) vector via Box-Muller and L2-normalises it. The same
 * text always maps to the same vector (so a self-query self-matches even after
 * quantization), while distinct texts map to near-orthogonal vectors (so
 * semantic-ordering assertions are stable). Implements the EmbeddingsInterface
 * shape (`embedDocuments` / `embedQuery`).
 */
export class HashEmbeddings {
  constructor(private readonly dim: number = 64) {}

  private embed(text: string): number[] {
    // FNV-1a string hash -> 32-bit seed.
    let h = 0x811c9dc5;
    for (let i = 0; i < text.length; i++) {
      h ^= text.charCodeAt(i);
      h = Math.imul(h, 0x01000193);
    }
    const rng = mulberry32(h >>> 0);
    const v = new Array<number>(this.dim);
    for (let j = 0; j < this.dim; j++) {
      const u1 = Math.max(rng(), 1e-10);
      const u2 = rng();
      v[j] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    let norm = 0;
    // `j` is bounded by `this.dim`, the array length, so every access is in-bounds.
    for (let j = 0; j < this.dim; j++) norm += v[j]! * v[j]!;
    norm = Math.sqrt(norm) + 1e-9;
    for (let j = 0; j < this.dim; j++) v[j]! /= norm;
    return v;
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    return texts.map((t) => this.embed(t));
  }

  async embedQuery(text: string): Promise<number[]> {
    return this.embed(text);
  }
}

/** Assert two Float32Arrays are equal within a tolerance. */
export function assertClose(a: Float32Array, b: Float32Array, tol = 1e-5): void {
  if (a.length !== b.length) throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    // `i` is bounded by `a.length`; lengths are checked equal above.
    if (Math.abs(a[i]! - b[i]!) > tol) {
      throw new Error(`Values differ at [${i}]: ${a[i]} vs ${b[i]} (tol=${tol})`);
    }
  }
}
