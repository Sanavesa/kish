# Performance History

Optimization journey for the kish Turkish Draughts engine — from 1.7M to 365M nodes/sec (215x speedup).

**Benchmark**: Perft depth 7 from initial position (10,782,382 leaf nodes at the time; 10,782,308 since v1.1.8 generates duplicate capture actions only once)

## v1.0.0 — 365M nodes/sec (32.3 ms)
- Perft optimizations: scratch buffers, bulk counting, ray masks, `count_actions()`
- Production-ready release

## v0.9.0 — 116M nodes/sec (101.9 ms)
- Full architecture rewrite

## v0.8.0 — 94M nodes/sec (126.0 ms)
- Remove redundant occupied checks
- `Option` → sentinel for `capture_index`
- `#[inline(always)]` on `gen_inner_fast`
- Enable LTO + `codegen-units=1`

## v0.7.0 — 90M nodes/sec (130.8 ms)
- Algorithmic optimizations: bulk capture detection
- Inline max length tracking
- Remove `status()` check in perft

## v0.6.0 — 70M nodes/sec (169.5 ms)
- Replace `Option<Action>` with sentinel value
- Pre-allocate `Vec`
- `#[inline(always)]` on hot paths

## v0.5.0 — 45M nodes/sec (262.4 ms)
- Const generics for `generate_king_captures_at::gen_inner`
- Precompute pawn move positions
- Const generics throughout action generation

## v0.4.0 — 39M nodes/sec (300.0 ms)
- Add turn to `Board`, team to `Action`
- Use transmute for `Square` to `u8` conversion
- Bug fix: king chained capture

## v0.3.0 — 33M nodes/sec (358.9 ms)
- `#[repr(u8)]` on `GameStatus`
- Remove needless variables

## v0.2.0 — 30M nodes/sec (389.4 ms)
- Bitboard implementation (14x speedup from v0.1)

## v0.1.0 — 2.1M nodes/sec (5.7 s)
- `#[repr(u8)]` on `Square`/`Team` enums
- `debug_assert!` instead of `assert!`
- While loop instead of range iteration

## v0.0.1 — 1.7M nodes/sec (6.9 s)
- Original implementation
