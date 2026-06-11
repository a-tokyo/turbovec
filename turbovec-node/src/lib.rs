mod error;
pub mod id_map;
pub mod index;

// napi-rs v3 auto-registers all #[napi] items at link time via the
// __napi_register__ symbols emitted by the napi-derive macro.
// No manual module registration function is needed (unlike PyO3).
