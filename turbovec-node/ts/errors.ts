/**
 * Typed view of the native error codes turbovec attaches to thrown `Error`s.
 *
 * Every error raised by the native addon carries a stable string `.code`
 * (the napi runtime maps it onto the JS `Error.code` property). This module
 * exposes that set as a discriminated union plus a type guard so callers can
 * branch on it with full type-safety:
 *
 * ```ts
 * try {
 *   index.addWithIds(flat, handles, dim);
 * } catch (e) {
 *   if (isTurbovecError(e) && e.code === 'DIM_MISMATCH') { ... }
 * }
 * ```
 *
 * The union is kept in lockstep with `src/error.rs`. `GENERIC_FAILURE` is the
 * internal napi fallback (emitted only when a non-typed `napi::Status` is
 * surfaced) and is intentionally NOT part of the public union — code carried by
 * a deliberately-thrown turbovec error is always one of the named variants.
 */

/** Stable `.code` values attached to native turbovec errors (see `src/error.rs`). */
export type TurbovecErrorCode =
  | 'DIM_MISMATCH'
  | 'DIM_NOT_MULTIPLE_OF_8'
  | 'VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM'
  | 'IDS_COUNT_MISMATCH'
  | 'ID_ALREADY_PRESENT'
  | 'INVALID_INPUT_VALUE'
  | 'BIT_WIDTH_OUT_OF_RANGE'
  | 'DIM_NOT_POSITIVE_MULTIPLE_OF_8'
  | 'DIM_TOO_LARGE'
  | 'QUERY_DIM_MISMATCH'
  | 'MASK_LENGTH_MISMATCH'
  | 'ALLOWLIST_EMPTY'
  | 'ALLOWLIST_UNKNOWN_ID'
  | 'INDEX_OUT_OF_RANGE'
  | 'DIM_REQUIRED'
  | 'IO_ERROR'
  | 'INVALID_ARGUMENT';

const TURBOVEC_ERROR_CODES = new Set<string>([
  'DIM_MISMATCH',
  'DIM_NOT_MULTIPLE_OF_8',
  'VECTOR_BUFFER_NOT_MULTIPLE_OF_DIM',
  'IDS_COUNT_MISMATCH',
  'ID_ALREADY_PRESENT',
  'INVALID_INPUT_VALUE',
  'BIT_WIDTH_OUT_OF_RANGE',
  'DIM_NOT_POSITIVE_MULTIPLE_OF_8',
  'DIM_TOO_LARGE',
  'QUERY_DIM_MISMATCH',
  'MASK_LENGTH_MISMATCH',
  'ALLOWLIST_EMPTY',
  'ALLOWLIST_UNKNOWN_ID',
  'INDEX_OUT_OF_RANGE',
  'DIM_REQUIRED',
  'IO_ERROR',
  'INVALID_ARGUMENT',
] satisfies TurbovecErrorCode[]);

/**
 * Narrow an unknown thrown value to a native turbovec error — an `Error`
 * whose `.code` is one of the known {@link TurbovecErrorCode} variants.
 */
export function isTurbovecError(e: unknown): e is Error & { code: TurbovecErrorCode } {
  return (
    e instanceof Error &&
    'code' in e &&
    typeof (e as { code: unknown }).code === 'string' &&
    TURBOVEC_ERROR_CODES.has((e as { code: string }).code)
  );
}
