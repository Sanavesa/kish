//! Game tree search utilities for Turkish Draughts.
//!
//! This module provides perft functions for move generation testing:
//! - [`Board::perft`] - Sequential perft (fast for shallow depths)
//! - [`Board::perft_parallel`] - Parallel perft with transposition table (for deep searches)
//!
//! # Perft (Performance Test)
//!
//! The `perft` function counts leaf nodes at a given depth, which is the standard
//! method for verifying move generation correctness:
//!
//! ```rust
//! use kish::Board;
//!
//! let board = Board::new_default();
//!
//! // Depth 1: Count immediate legal moves
//! let nodes = board.perft(1);
//! println!("Legal moves from start: {}", nodes);
//!
//! // Depth 2: All responses to all first moves
//! let nodes = board.perft(2);
//! println!("Positions after 2 plies: {}", nodes);
//! ```
//!
//! # Performance
//!
//! The sequential implementation uses:
//! - XOR-based state updates for efficient apply/undo during tree traversal
//! - Scratch buffer reuse to eliminate Vec allocations in the hot path
//! - Bulk leaf counting at depth 1 to avoid unnecessary recursion
//!
//! The parallel implementation adds:
//! - Rayon for parallel traversal at top levels
//! - Lock-free transposition table using DashMap for position caching

use super::{Action, Board};
use rayon::prelude::*;
use rustc_hash::FxHasher;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

impl Board {
    /// Perft (performance test) - counts leaf nodes at a given depth.
    ///
    /// This is the standard metric for verifying move generation correctness.
    /// Each unique position at the target depth is counted exactly once.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kish::Board;
    ///
    /// let board = Board::new_default();
    /// let nodes = board.perft(3);
    /// println!("Positions at depth 3: {}", nodes);
    /// ```
    #[must_use]
    pub fn perft(&self, depth: u64) -> u64 {
        if depth == 0 {
            return 1;
        }

        // Pre-allocate scratch buffers:
        // - One for depth-1 counting (shared across all depth-1 calls)
        // - One per depth level for depths 2+
        // Capacity 48 covers most positions (typical max is ~40 actions)
        let mut count_scratch = Vec::with_capacity(48);
        let mut scratches: Vec<Vec<Action>> = (1..depth).map(|_| Vec::with_capacity(48)).collect();
        self.perft_inner(depth, &mut scratches, &mut count_scratch)
    }

    /// Internal perft implementation with per-level scratch buffer reuse.
    ///
    /// Each recursion level has its own scratch buffer to avoid overwrites.
    /// Buffers are pre-allocated once and reused throughout the search.
    #[inline(always)]
    fn perft_inner(
        &self,
        depth: u64,
        scratches: &mut [Vec<Action>],
        count_scratch: &mut Vec<Action>,
    ) -> u64 {
        // Bulk leaf optimization: at depth 1, count actions without generating them
        if depth == 1 {
            let count = self.count_actions(count_scratch);
            return if count == 0 { 1 } else { count };
        }

        // Use depth-2 as index since we skip depth 1 buffer
        // (depth 7 uses index 5, depth 2 uses index 0)
        let idx = depth as usize - 2;

        self.actions_into(&mut scratches[idx]);
        if scratches[idx].is_empty() {
            return 1; // Terminal node counts as 1
        }

        let action_count = scratches[idx].len();
        let mut nodes = 0u64;
        for i in 0..action_count {
            let action = scratches[idx][i];
            let mut board = self.apply(&action);
            board.swap_turn_();
            nodes += board.perft_inner(depth - 1, scratches, count_scratch);
        }
        nodes
    }

    /// Parallel perft with transposition table for deep searches.
    ///
    /// Uses rayon to parallelize at the top level and a lock-free transposition
    /// table to cache and reuse results for positions that occur via transposition.
    ///
    /// # Arguments
    /// * `depth` - The search depth
    /// * `tt_size_mb` - Approximate size of transposition table in megabytes (0 to disable)
    ///
    /// # Example
    ///
    /// ```rust
    /// use kish::Board;
    ///
    /// let board = Board::new_default();
    /// // Use 256MB transposition table
    /// let nodes = board.perft_parallel(9, 256);
    /// println!("Perft(9) = {}", nodes);
    /// ```
    #[must_use]
    pub fn perft_parallel(&self, depth: u64, tt_size_mb: usize) -> u64 {
        if depth == 0 {
            return 1;
        }
        if depth <= 2 {
            return self.perft(depth);
        }

        // Calculate TT capacity based on size
        // Each entry: Board (32 bytes key via hash) + u64 (8 bytes value) + u8 (1 byte depth)
        // With overhead, estimate ~64 bytes per entry
        let tt_capacity = if tt_size_mb > 0 {
            (tt_size_mb * 1024 * 1024) / 64
        } else {
            0
        };

        // Create shared transposition table
        let tt = TranspositionTable::new(tt_capacity);
        let tt_hits = AtomicU64::new(0);
        let tt_lookups = AtomicU64::new(0);

        // Generate first-level actions
        let actions = self.actions();
        if actions.is_empty() {
            return 1;
        }

        // Parallel search at top level
        let nodes: u64 = actions
            .par_iter()
            .map(|action| {
                let mut board = self.apply(action);
                board.swap_turn_();

                // Each thread gets its own scratch buffers
                let mut count_scratch = Vec::with_capacity(48);
                let mut scratches: Vec<Vec<Action>> =
                    (1..depth).map(|_| Vec::with_capacity(48)).collect();

                board.perft_tt_inner(
                    depth - 1,
                    &mut scratches,
                    &mut count_scratch,
                    &tt,
                    &tt_hits,
                    &tt_lookups,
                )
            })
            .sum();

        // Print TT hit statistics
        let hits = tt_hits.load(Ordering::Relaxed);
        let lookups = tt_lookups.load(Ordering::Relaxed);
        if lookups > 0 {
            let hit_rate = (hits as f64 / lookups as f64) * 100.0;
            eprintln!(
                "TT hits: {} / {} lookups ({:.2}% hit rate)",
                hits, lookups, hit_rate
            );
        }

        nodes
    }

    /// Internal perft with transposition table lookup.
    #[inline(always)]
    fn perft_tt_inner(
        &self,
        depth: u64,
        scratches: &mut [Vec<Action>],
        count_scratch: &mut Vec<Action>,
        tt: &TranspositionTable,
        tt_hits: &AtomicU64,
        tt_lookups: &AtomicU64,
    ) -> u64 {
        // Bulk leaf optimization
        if depth == 1 {
            let count = self.count_actions(count_scratch);
            return if count == 0 { 1 } else { count };
        }

        // TT lookup (only for depth >= 3 to avoid overhead)
        if depth >= 3 {
            tt_lookups.fetch_add(1, Ordering::Relaxed);
            if let Some(nodes) = tt.get(self, depth as u8) {
                tt_hits.fetch_add(1, Ordering::Relaxed);
                return nodes;
            }
        }

        let idx = depth as usize - 2;
        self.actions_into(&mut scratches[idx]);
        if scratches[idx].is_empty() {
            return 1;
        }

        let action_count = scratches[idx].len();
        let mut nodes = 0u64;
        for i in 0..action_count {
            let action = scratches[idx][i];
            let mut board = self.apply(&action);
            board.swap_turn_();
            nodes +=
                board.perft_tt_inner(depth - 1, scratches, count_scratch, tt, tt_hits, tt_lookups);
        }

        // Store in TT (only for depth >= 3)
        if depth >= 3 {
            tt.insert(self, depth as u8, nodes);
        }

        nodes
    }
}

/// Lock-free transposition table using a simple hash table with replacement.
///
/// Uses Zobrist-style hashing where collisions are handled by replacement.
/// This is acceptable for perft since we're counting, not searching for best moves.
struct TranspositionTable {
    /// Table entries: (hash_verification, depth, nodes)
    /// We store a verification hash to detect collisions
    entries: Vec<AtomicEntry>,
    mask: usize,
}

/// Atomic entry for lock-free access
struct AtomicEntry {
    /// Combined: upper 56 bits = hash verification, lower 8 bits = depth
    key: AtomicU64,
    /// Node count
    value: AtomicU64,
}

impl TranspositionTable {
    fn new(capacity: usize) -> Self {
        if capacity == 0 {
            return Self {
                entries: Vec::new(),
                mask: 0,
            };
        }

        // Round up to power of 2
        let capacity = capacity.next_power_of_two();
        let mask = capacity - 1;

        let mut entries = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            entries.push(AtomicEntry {
                key: AtomicU64::new(0),
                value: AtomicU64::new(0),
            });
        }

        Self { entries, mask }
    }

    /// XOR trick for lockless hashing: by XORing key with value on store,
    /// we can detect torn reads where key and value come from different writes.
    #[inline]
    fn get(&self, board: &Board, depth: u8) -> Option<u64> {
        if self.entries.is_empty() {
            return None;
        }

        let hash = self.hash_board(board);
        let index = (hash as usize) & self.mask;
        let entry = &self.entries[index];

        // Load value first, then key (order matters for XOR trick)
        let value = entry.value.load(Ordering::Relaxed);
        let stored_key_xored = entry.key.load(Ordering::Relaxed);

        // Recover original key by XORing with value
        let recovered_key = stored_key_xored ^ value;
        let expected_key = (hash & 0xFFFF_FFFF_FFFF_FF00) | (depth as u64);

        if recovered_key == expected_key {
            Some(value)
        } else {
            None
        }
    }

    #[inline]
    fn insert(&self, board: &Board, depth: u8, nodes: u64) {
        if self.entries.is_empty() {
            return;
        }

        let hash = self.hash_board(board);
        let index = (hash as usize) & self.mask;
        let entry = &self.entries[index];

        let key = (hash & 0xFFFF_FFFF_FFFF_FF00) | (depth as u64);
        // XOR trick: store key ^ value so torn reads are detected
        entry.key.store(key ^ nodes, Ordering::Relaxed);
        entry.value.store(nodes, Ordering::Relaxed);
    }

    #[inline]
    fn hash_board(&self, board: &Board) -> u64 {
        let build_hasher = BuildHasherDefault::<FxHasher>::default();
        let mut hasher = build_hasher.build_hasher();
        board.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Square, Team};

    // ========== Perft Regression Tests ==========
    // These verify move generation correctness by comparing against known values

    #[test]
    fn perft_depth_0() {
        let board = Board::new_default();
        assert_eq!(board.perft(0), 1);
    }

    #[test]
    fn perft_depth_1() {
        let board = Board::new_default();
        assert_eq!(board.perft(1), 8);
    }

    #[test]
    fn perft_depth_2() {
        let board = Board::new_default();
        assert_eq!(board.perft(2), 64);
    }

    #[test]
    fn perft_depth_3() {
        let board = Board::new_default();
        assert_eq!(board.perft(3), 708);
    }

    #[test]
    fn perft_depth_4() {
        let board = Board::new_default();
        assert_eq!(board.perft(4), 7538);
    }

    #[test]
    fn perft_depth_5() {
        let board = Board::new_default();
        assert_eq!(board.perft(5), 85090);
    }

    #[test]
    fn perft_depth_6() {
        let board = Board::new_default();
        assert_eq!(board.perft(6), 931_312);
    }

    #[test]
    fn perft_depth_7() {
        let board = Board::new_default();
        assert_eq!(board.perft(7), 10_782_382);
    }

    #[test]
    fn perft_depth_8() {
        let board = Board::new_default();
        assert_eq!(board.perft(8), 123_290_300);
    }

    #[test]
    fn perft_parallel_matches_sequential() {
        let board = Board::new_default();
        // Test at depth 7 where we know the answer
        let seq = board.perft(7);
        let par = board.perft_parallel(7, 64); // 64MB TT
        assert_eq!(seq, par);
    }

    // ========== Simple Position Perft Tests ==========

    #[test]
    fn perft_single_pawn() {
        // Single white pawn at D4
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[]);
        let perft1 = board.perft(1);
        // D4 pawn can move to C4, E4, D5 = 3 moves
        assert_eq!(perft1, 3, "Single pawn at D4 should have 3 moves");
    }

    #[test]
    fn perft_single_king() {
        // Single white king at D4
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[Square::D4]);
        let perft1 = board.perft(1);
        // King at D4: left(3) + right(4) + up(4) + down(3) = 14 moves
        assert_eq!(perft1, 14, "King at D4 should have 14 moves");
    }

    #[test]
    fn perft_forced_capture() {
        // White pawn at D4, black pawn at D5 (capturable)
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5], &[]);
        let perft1 = board.perft(1);
        // Only capture available
        assert_eq!(perft1, 1, "Should have exactly 1 capture");
    }

    #[test]
    fn perft_terminal_position() {
        // White has no pieces - game over
        let board = Board::from_squares(Team::White, &[], &[Square::D4], &[]);
        let perft1 = board.perft(1);
        assert_eq!(perft1, 1, "Terminal position should count as 1");
    }
}
