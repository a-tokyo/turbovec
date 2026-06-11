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
    for (let j = 0; j < dim; j++) norm += row[j] * row[j];
    norm = Math.sqrt(norm) + 1e-9;
    for (let j = 0; j < dim; j++) row[j] /= norm;
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
  return buf.slice(i * k, (i + 1) * k) as Float32Array | BigInt64Array | BigUint64Array;
}

/** Extract a row of Float32Array values as a regular JS number array. */
export function rowNumbers(buf: Float32Array, i: number, k: number): number[] {
  return Array.from(row(buf, i, k) as Float32Array);
}

/** Assert two Float32Arrays are equal within a tolerance. */
export function assertClose(a: Float32Array, b: Float32Array, tol = 1e-5): void {
  if (a.length !== b.length) throw new Error(`Length mismatch: ${a.length} vs ${b.length}`);
  for (let i = 0; i < a.length; i++) {
    if (Math.abs(a[i] - b[i]) > tol) {
      throw new Error(`Values differ at [${i}]: ${a[i]} vs ${b[i]} (tol=${tol})`);
    }
  }
}
