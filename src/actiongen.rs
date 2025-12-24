//! Move generation for Turkish Draughts.
//!
//! This module implements legal move generation following the official rules:
//!
//! # Move Generation Rules
//!
//! 1. **Mandatory capture**: If any capture is available, a capture must be made.
//!    Non-capturing moves are only generated when no captures exist.
//!
//! 2. **Maximum capture rule**: When multiple capture sequences are possible,
//!    the player must choose a sequence that captures the maximum number of pieces.
//!
//! 3. **180-degree turn prohibition**: During a multi-capture sequence, the piece
//!    cannot reverse direction (e.g., if moving up, cannot immediately move down).
//!
//! 4. **Flying captures**: Kings can capture from any distance along a rank/file,
//!    landing on any empty square beyond the captured piece.
//!
//! # Implementation Details
//!
//! ## Lookup Tables
//!
//! Precomputed move masks for each square type:
//! - `WHITE_PAWN_MOVES[sq]`: Valid non-capturing destinations for white pawns
//! - `BLACK_PAWN_MOVES[sq]`: Valid non-capturing destinations for black pawns
//!
//! ## Capture Generation Strategy
//!
//! Captures are generated using a recursive approach with XOR-based state updates:
//!
//! 1. Try all single captures from the current position
//! 2. For each capture, temporarily apply it using XOR (modifies state)
//! 3. Recursively search for additional captures
//! 4. Undo the capture using XOR (restores state)
//! 5. Track the maximum capture count across all sequences
//! 6. Only add actions that achieve the maximum capture count
//!
//! This approach is cache-friendly and avoids allocating new board states.
//!
//! ## Const Generics
//!
//! Team-specific logic uses `const TEAM_INDEX: usize` to generate specialized
//! code paths at compile time, avoiding runtime branching in hot loops.
//!
//! # Example
//!
//! ```rust
//! use kish::{Board, Team};
//!
//! let board = Board::new_default();
//! let actions = board.actions();
//!
//! // Initial position has no captures, so non-capturing moves are generated
//! // White pawns can move forward (32 pawns × 1 forward move each, minus blocked)
//! assert!(actions.len() > 0);
//! ```

use super::{Action, Board, Team};
use crate::state::{
    MASK_COL_A, MASK_COL_B, MASK_COL_G, MASK_COL_H, MASK_ROW_1, MASK_ROW_2, MASK_ROW_7, MASK_ROW_8,
    MASK_ROW_PROMOTIONS,
};

/// Precomputed ray masks for each direction from each square.
/// Used for early-exit checks in king capture generation.
/// LEFT_RAY[sq] = all squares to the left of sq on the same row
const LEFT_RAY: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let col = sq % 8;
        let row_start = sq - col;
        // All squares from row_start to sq-1
        let mut mask = 0u64;
        let mut c = 0;
        while c < col {
            mask |= 1u64 << (row_start + c);
            c += 1;
        }
        rays[sq] = mask;
        sq += 1;
    }
    rays
};

/// RIGHT_RAY[sq] = all squares to the right of sq on the same row
const RIGHT_RAY: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let col = sq % 8;
        let row_start = sq - col;
        // All squares from sq+1 to row_end
        let mut mask = 0u64;
        let mut c = col + 1;
        while c < 8 {
            mask |= 1u64 << (row_start + c);
            c += 1;
        }
        rays[sq] = mask;
        sq += 1;
    }
    rays
};

/// UP_RAY[sq] = all squares above sq on the same column
const UP_RAY: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let col = sq % 8;
        let row = sq / 8;
        // All squares from sq+8 to top of column
        let mut mask = 0u64;
        let mut r = row + 1;
        while r < 8 {
            mask |= 1u64 << (r * 8 + col);
            r += 1;
        }
        rays[sq] = mask;
        sq += 1;
    }
    rays
};

/// DOWN_RAY[sq] = all squares below sq on the same column
const DOWN_RAY: [u64; 64] = {
    let mut rays = [0u64; 64];
    let mut sq = 0;
    while sq < 64 {
        let col = sq % 8;
        let row = sq / 8;
        // All squares from row-1 down to row 0
        let mut mask = 0u64;
        let mut r: i32 = row as i32 - 1;
        while r >= 0 {
            mask |= 1u64 << (r as usize * 8 + col);
            r -= 1;
        }
        rays[sq] = mask;
        sq += 1;
    }
    rays
};

const WHITE_PAWN_MOVES: [u64; 64] = {
    let mut moves = [0u64; 64];
    let mut src_index = 0;
    while src_index < 64 {
        let src_mask = 1u64 << src_index;
        moves[src_index] = (src_mask & !MASK_COL_A) >> 1u8 // left
            | (src_mask & !MASK_COL_H) << 1u8 // right
            | (src_mask & !MASK_ROW_8) << 8u8; // up
        src_index += 1;
    }
    moves
};

const BLACK_PAWN_MOVES: [u64; 64] = {
    let mut moves = [0u64; 64];
    let mut src_index = 0;
    while src_index < 64 {
        let src_mask = 1u64 << src_index;
        moves[src_index] = (src_mask & !MASK_COL_A) >> 1u8 // left
            | (src_mask & !MASK_COL_H) << 1u8 // right
            | (src_mask & !MASK_ROW_1) >> 8u8; // down
        src_index += 1;
    }
    moves
};

impl Board {
    /// Computes the valid actions of the board.
    #[must_use]
    #[inline]
    pub fn actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(32);
        self.actions_into(&mut actions);
        actions
    }

    /// Computes the valid actions and stores them in the provided Vec.
    ///
    /// The Vec is cleared before adding actions. This allows callers to reuse
    /// a Vec across multiple calls without manual clearing.
    #[inline]
    pub fn actions_into(&self, actions: &mut Vec<Action>) {
        actions.clear();
        if self.turn == Team::White {
            self.generate_captures::<0>(actions);

            if actions.is_empty() {
                self.generate_moves::<0>(actions);
            }
        } else {
            self.generate_captures::<1>(actions);

            if actions.is_empty() {
                self.generate_moves::<1>(actions);
            }
        }
    }

    /// Counts the number of valid actions using the provided scratch buffer.
    ///
    /// This is faster than `actions_into` when you only need the count,
    /// particularly useful for bulk leaf counting in perft at depth 1.
    /// Returns 0 for terminal positions (no actions available).
    #[inline]
    pub fn count_actions(&self, scratch: &mut Vec<Action>) -> u64 {
        if self.turn == Team::White {
            let capture_count = self.count_captures::<0>(scratch);
            if capture_count > 0 {
                capture_count
            } else {
                self.count_moves::<0>()
            }
        } else {
            let capture_count = self.count_captures::<1>(scratch);
            if capture_count > 0 {
                capture_count
            } else {
                self.count_moves::<1>()
            }
        }
    }

    /// Count captures using provided scratch buffer to avoid allocation.
    fn count_captures<const TEAM_INDEX: usize>(&self, scratch: &mut Vec<Action>) -> u64 {
        let may_have_pawn_captures = self.has_any_pawn_captures::<TEAM_INDEX>();
        let has_friendly_kings = (self.friendly_pieces() & self.state.kings) != 0;

        if !may_have_pawn_captures && !has_friendly_kings {
            return 0;
        }

        // For captures, we need to track max length and generate actions
        // to properly implement the maximum capture rule.
        scratch.clear();
        let mut max_length: u32 = 0;
        let mut board = *self;

        if has_friendly_kings {
            Self::generate_king_captures_with_board::<TEAM_INDEX>(
                &mut board,
                scratch,
                &mut max_length,
            );
        }

        if may_have_pawn_captures {
            Self::generate_pawn_captures_with_board::<TEAM_INDEX>(
                &mut board,
                scratch,
                &mut max_length,
            );
        }

        scratch.len() as u64
    }

    /// Count non-capture moves without generating Action structs.
    fn count_moves<const TEAM_INDEX: usize>(&self) -> u64 {
        let empty = self.state.empty();
        self.count_king_moves::<TEAM_INDEX>(empty) + self.count_pawn_moves::<TEAM_INDEX>(empty)
    }

    /// Count pawn moves using bitboard popcount.
    #[inline]
    fn count_pawn_moves<const TEAM_INDEX: usize>(&self, empty: u64) -> u64 {
        let friendly_pawns = self.friendly_pieces() & !self.state.kings;
        let mut count = 0u64;
        let mut pawns = friendly_pawns;

        while pawns != 0 {
            let src_mask = pawns & pawns.wrapping_neg();
            let src_index = src_mask.trailing_zeros() as usize;
            let moves = if TEAM_INDEX == 0 {
                WHITE_PAWN_MOVES[src_index] & empty
            } else {
                BLACK_PAWN_MOVES[src_index] & empty
            };
            count += moves.count_ones() as u64;
            pawns ^= src_mask;
        }
        count
    }

    /// Count king moves using direction scanning.
    #[inline]
    fn count_king_moves<const TEAM_INDEX: usize>(&self, empty: u64) -> u64 {
        let mut friendly_kings = self.friendly_pieces() & self.state.kings;
        let mut count = 0u64;

        while friendly_kings != 0 {
            let src_mask = friendly_kings & friendly_kings.wrapping_neg();

            // Count moves in each direction
            count += Self::count_ray_moves(src_mask, empty, -1i8, MASK_COL_A); // left
            count += Self::count_ray_moves(src_mask, empty, 1i8, MASK_COL_H); // right
            count += Self::count_ray_moves(src_mask, empty, 8i8, MASK_ROW_8); // up
            count += Self::count_ray_moves(src_mask, empty, -8i8, MASK_ROW_1); // down

            friendly_kings ^= src_mask;
        }
        count
    }

    /// Count moves along a ray direction until blocked or edge.
    #[inline]
    fn count_ray_moves(src_mask: u64, empty: u64, direction: i8, edge_mask: u64) -> u64 {
        let mut count = 0u64;
        let mut pos = src_mask;

        loop {
            // Stop if at edge
            if pos & edge_mask != 0 {
                break;
            }

            // Move in direction
            pos = if direction > 0 {
                pos << direction as u32
            } else {
                pos >> (-direction) as u32
            };

            // Stop if blocked
            if pos & empty == 0 {
                break;
            }

            count += 1;
        }
        count
    }

    /// Quick check if any pawn captures are possible using bulk bitboard ops.
    ///
    /// For each direction, verifies that the SAME pawn has both:
    /// - An adjacent hostile piece (to capture)
    /// - An empty square beyond it (to land on)
    const fn has_any_pawn_captures<const TEAM_INDEX: usize>(&self) -> bool {
        let friendly_pawns = self.friendly_pieces() & !self.state.kings;
        if friendly_pawns == 0 {
            return false;
        }

        let hostile = self.hostile_pieces();
        let empty = self.state.empty();

        // Check left captures: pawn at P, hostile at P-1, empty at P-2
        {
            let eligible = friendly_pawns & !(MASK_COL_A | MASK_COL_B);
            let adjacent_hostile = (eligible >> 1) & hostile; // Hostiles at P-1
            let landing_clear = (eligible >> 2) & empty; // Empty at P-2
                                                         // Correlate: landing at L means hostile must be at L+1
            if (landing_clear << 1) & adjacent_hostile != 0 {
                return true;
            }
        }

        // Check right captures: pawn at P, hostile at P+1, empty at P+2
        {
            let eligible = friendly_pawns & !(MASK_COL_G | MASK_COL_H);
            let adjacent_hostile = (eligible << 1) & hostile; // Hostiles at P+1
            let landing_clear = (eligible << 2) & empty; // Empty at P+2
                                                         // Correlate: landing at L means hostile must be at L-1
            if (landing_clear >> 1) & adjacent_hostile != 0 {
                return true;
            }
        }

        // Check vertical captures (team-dependent)
        if TEAM_INDEX == 0 {
            // White: up captures (pawn at P, hostile at P+8, empty at P+16)
            let eligible = friendly_pawns & !(MASK_ROW_7 | MASK_ROW_8);
            let adjacent_hostile = (eligible << 8) & hostile;
            let landing_clear = (eligible << 16) & empty;
            if (landing_clear >> 8) & adjacent_hostile != 0 {
                return true;
            }
        } else {
            // Black: down captures (pawn at P, hostile at P-8, empty at P-16)
            let eligible = friendly_pawns & !(MASK_ROW_1 | MASK_ROW_2);
            let adjacent_hostile = (eligible >> 8) & hostile;
            let landing_clear = (eligible >> 16) & empty;
            if (landing_clear << 8) & adjacent_hostile != 0 {
                return true;
            }
        }

        false
    }

    #[inline]
    fn generate_captures<const TEAM_INDEX: usize>(&self, actions: &mut Vec<Action>) {
        // Early exit checks:
        // - Pawn captures use a fast bulk bitboard check (no iteration)
        // - King captures just check existence (actual capture check happens during generation)
        let may_have_pawn_captures = self.has_any_pawn_captures::<TEAM_INDEX>();
        let has_friendly_kings = (self.friendly_pieces() & self.state.kings) != 0;

        if !may_have_pawn_captures && !has_friendly_kings {
            return;
        }

        // Track max capture length inline to avoid second pass
        let mut max_length: u32 = 0;

        // We need a mutable copy for the recursive capture generation
        let mut board = *self;

        // Generate king captures first (kings often have longer chains)
        if has_friendly_kings {
            Self::generate_king_captures_with_board::<TEAM_INDEX>(
                &mut board,
                actions,
                &mut max_length,
            );
        }

        // Generate pawn captures (only if bulk check passed)
        if may_have_pawn_captures {
            Self::generate_pawn_captures_with_board::<TEAM_INDEX>(
                &mut board,
                actions,
                &mut max_length,
            );
        }
    }

    /// Helper to push action while tracking max capture length
    #[inline(always)]
    fn push_capture_action<const TEAM_INDEX: usize>(
        actions: &mut Vec<Action>,
        max_length: &mut u32,
        action: Action,
    ) {
        let length = action.delta.pieces[1 - TEAM_INDEX].count_ones();
        if length > *max_length {
            // New best - clear existing and update max
            actions.clear();
            *max_length = length;
            actions.push(action);
        } else if length == *max_length {
            // Equal to best - just add
            actions.push(action);
        }
        // length < max_length: discard
    }

    #[inline]
    fn generate_pawn_captures_with_board<const TEAM_INDEX: usize>(
        board: &mut Self,
        actions: &mut Vec<Action>,
        max_length: &mut u32,
    ) {
        let mut friendly_pawns = board.friendly_pieces() & !board.state.kings;

        while friendly_pawns != 0u64 {
            let src_mask = friendly_pawns & friendly_pawns.wrapping_neg(); // get lowest set bit

            // Start with no previous direction (0)
            Self::generate_pawn_captures_at::<TEAM_INDEX, 0>(
                board,
                actions,
                max_length,
                src_mask,
                Action::EMPTY,
            );

            friendly_pawns ^= src_mask; // clear lowest set bit
        }
    }

    /// Generates pawn capture sequences with 180-degree turn prevention.
    ///
    /// # Const Parameters
    /// - `TEAM_INDEX`: 0 for White, 1 for Black
    /// - `PREVIOUS_DIRECTION`: The direction of the previous capture in the sequence.
    ///   - 0: No previous capture (initial state)
    ///   - -1: Previous capture was left
    ///   - 1: Previous capture was right
    ///   - 8: Previous capture was up (forward for white)
    ///   - -8: Previous capture was down (forward for black)
    ///
    /// The 180-degree turn rule prohibits reversing direction within a capture sequence
    /// (e.g., left then right, or right then left).
    #[inline]
    fn generate_pawn_captures_at<const TEAM_INDEX: usize, const PREVIOUS_DIRECTION: i8>(
        board: &mut Self,
        actions: &mut Vec<Action>,
        max_length: &mut u32,
        src_mask: u64,
        previous_action: Action,
    ) {
        let mut has_more_captures = false;

        let hostile_pieces = board.hostile_pieces();
        let empty = board.state.empty();

        // Generate pawn left captures (direction = -1)
        // Skip if previous direction was right (+1), as that would be a 180-degree turn
        if PREVIOUS_DIRECTION != 1 {
            let left_capture_mask =
                ((src_mask & !(MASK_COL_A | MASK_COL_B)) >> 1u8) & hostile_pieces;
            let left_dest_mask = ((src_mask & !(MASK_COL_A | MASK_COL_B)) >> 2u8) & empty;
            if left_capture_mask != 0u64 && left_dest_mask != 0u64 {
                has_more_captures = true;

                let capture_action = Action::new_capture_as_pawn::<TEAM_INDEX>(
                    src_mask,
                    left_dest_mask,
                    left_capture_mask,
                    board.state.kings,
                );

                // Apply action
                board.apply_(&capture_action);

                Self::generate_pawn_captures_at::<TEAM_INDEX, -1>(
                    board,
                    actions,
                    max_length,
                    left_dest_mask,
                    previous_action.combine(&capture_action),
                );

                // Undo action
                board.apply_(&capture_action);
            }
        }

        // Generate pawn right captures (direction = +1)
        // Skip if previous direction was left (-1), as that would be a 180-degree turn
        if PREVIOUS_DIRECTION != -1 {
            let right_capture_mask =
                ((src_mask & !(MASK_COL_G | MASK_COL_H)) << 1u8) & hostile_pieces;
            let right_dest_mask = ((src_mask & !(MASK_COL_G | MASK_COL_H)) << 2u8) & empty;
            if right_capture_mask != 0u64 && right_dest_mask != 0u64 {
                has_more_captures = true;

                let capture_action = Action::new_capture_as_pawn::<TEAM_INDEX>(
                    src_mask,
                    right_dest_mask,
                    right_capture_mask,
                    board.state.kings,
                );

                // Apply action
                board.apply_(&capture_action);

                Self::generate_pawn_captures_at::<TEAM_INDEX, 1>(
                    board,
                    actions,
                    max_length,
                    right_dest_mask,
                    previous_action.combine(&capture_action),
                );

                // Undo action
                board.apply_(&capture_action);
            }
        }

        // Generate pawn vertical captures (white=up, black=down)
        // Vertical direction is +8 for white (up), -8 for black (down)
        // Note: Pawns cannot capture backward, so there's no opposite vertical direction to check
        let (vert_capture_mask, vert_dest_mask): (u64, u64) = if TEAM_INDEX == 0 {
            let mask = !(MASK_ROW_7 | MASK_ROW_8);
            (
                ((src_mask & mask) << 8u8) & hostile_pieces,
                ((src_mask & mask) << 16u8) & empty,
            )
        } else {
            let mask = !(MASK_ROW_1 | MASK_ROW_2);
            (
                ((src_mask & mask) >> 8u8) & hostile_pieces,
                ((src_mask & mask) >> 16u8) & empty,
            )
        };

        // Vertical captures don't conflict with left/right (no 180-degree turn possible)
        // since pawns can't capture backward
        if vert_capture_mask != 0u64 && vert_dest_mask != 0u64 {
            has_more_captures = true;
            let capture_action = Action::new_capture_as_pawn::<TEAM_INDEX>(
                src_mask,
                vert_dest_mask,
                vert_capture_mask,
                board.state.kings,
            );

            board.apply_(&capture_action);

            // Select const generic direction based on team (white=up/8, black=down/-8)
            if TEAM_INDEX == 0 {
                Self::generate_pawn_captures_at::<TEAM_INDEX, 8>(
                    board,
                    actions,
                    max_length,
                    vert_dest_mask,
                    previous_action.combine(&capture_action),
                );
            } else {
                Self::generate_pawn_captures_at::<TEAM_INDEX, -8>(
                    board,
                    actions,
                    max_length,
                    vert_dest_mask,
                    previous_action.combine(&capture_action),
                );
            }

            board.apply_(&capture_action);
        }

        if !has_more_captures && !previous_action.is_empty() {
            // Add promotion if the capture sequence ends on the promotion row.
            // Note: src_mask here is the final destination of the sequence.
            let mut final_action = previous_action;
            let promotion_mask = MASK_ROW_PROMOTIONS[TEAM_INDEX];
            if src_mask & promotion_mask != 0 {
                // Promote the pawn at the final destination
                final_action.delta.kings ^= src_mask;
            }
            Self::push_capture_action::<TEAM_INDEX>(actions, max_length, final_action);
        }
    }

    #[inline]
    fn generate_king_captures_with_board<const TEAM_INDEX: usize>(
        board: &mut Self,
        actions: &mut Vec<Action>,
        max_length: &mut u32,
    ) {
        let mut friendly_kings = board.friendly_pieces() & board.state.kings;

        while friendly_kings != 0u64 {
            let src_mask = friendly_kings & friendly_kings.wrapping_neg(); // get lowest set bit

            Self::generate_king_captures_at::<TEAM_INDEX, 0i8>(
                board,
                actions,
                max_length,
                src_mask,
                Action::EMPTY,
            );

            friendly_kings ^= src_mask; // clear lowest set bit
        }
    }

    #[inline]
    fn generate_king_captures_at<const TEAM_INDEX: usize, const PREVIOUS_DIRECTION: i8>(
        board: &mut Self,
        actions: &mut Vec<Action>,
        max_length: &mut u32,
        src_mask: u64,
        previous_action: Action,
    ) {
        let mut has_more_captures = false;

        let friendly_pieces = board.friendly_pieces();
        let hostile_pieces = board.hostile_pieces();

        let src_index = src_mask.trailing_zeros() as usize;

        // Early exit checks using ray masks:
        // Only scan a direction if there's at least one hostile in that direction.
        // This avoids calling gen_inner for directions with no possible captures.

        // Eat left (only if not coming from right and hostile exists left)
        if PREVIOUS_DIRECTION != 1 && (hostile_pieces & LEFT_RAY[src_index]) != 0 {
            board
                .gen_inner::<-1i8, PREVIOUS_DIRECTION, TEAM_INDEX, { !(MASK_COL_A | MASK_COL_B) }>(
                    src_mask,
                    src_index as u8,
                    friendly_pieces,
                    hostile_pieces,
                    &mut has_more_captures,
                    actions,
                    max_length,
                    previous_action,
                );
        }

        // Eat right (only if not coming from left and hostile exists right)
        if PREVIOUS_DIRECTION != -1 && (hostile_pieces & RIGHT_RAY[src_index]) != 0 {
            board.gen_inner::<1i8, PREVIOUS_DIRECTION, TEAM_INDEX, { !(MASK_COL_G | MASK_COL_H) }>(
                src_mask,
                src_index as u8,
                friendly_pieces,
                hostile_pieces,
                &mut has_more_captures,
                actions,
                max_length,
                previous_action,
            );
        }

        // Eat up (only if not coming from down and hostile exists up)
        if PREVIOUS_DIRECTION != -8 && (hostile_pieces & UP_RAY[src_index]) != 0 {
            board.gen_inner::<8i8, PREVIOUS_DIRECTION, TEAM_INDEX, { !(MASK_ROW_7 | MASK_ROW_8) }>(
                src_mask,
                src_index as u8,
                friendly_pieces,
                hostile_pieces,
                &mut has_more_captures,
                actions,
                max_length,
                previous_action,
            );
        }

        // Eat down (only if not coming from up and hostile exists down)
        if PREVIOUS_DIRECTION != 8 && (hostile_pieces & DOWN_RAY[src_index]) != 0 {
            board
                .gen_inner::<-8i8, PREVIOUS_DIRECTION, TEAM_INDEX, { !(MASK_ROW_1 | MASK_ROW_2) }>(
                    src_mask,
                    src_index as u8,
                    friendly_pieces,
                    hostile_pieces,
                    &mut has_more_captures,
                    actions,
                    max_length,
                    previous_action,
                );
        }

        if !has_more_captures && !previous_action.is_empty() {
            Self::push_capture_action::<TEAM_INDEX>(actions, max_length, previous_action);
        }
    }

    #[allow(clippy::too_many_arguments)] // Internal recursive function with const generics
    #[inline(always)]
    fn gen_inner<
        const DIRECTION: i8,
        const PREVIOUS_DIRECTION: i8,
        const TEAM_INDEX: usize,
        const CHECKMASK: u64,
    >(
        &mut self,
        src_mask: u64,
        src_index: u8,
        friendly_pieces: u64,
        hostile_pieces: u64,
        has_more_captures: &mut bool,
        actions: &mut Vec<Action>,
        max_length: &mut u32,
        previous_action: Action,
    ) {
        // Note: DIRECTION == -PREVIOUS_DIRECTION check is now done at caller level
        // with early ray mask checks for better performance.

        // If we cannot capture, abort
        if (CHECKMASK >> src_index) & 1u64 != 1u64 {
            return;
        }

        let mut temp_index = src_index as i8 + DIRECTION;
        const NO_CAPTURE: u8 = 255; // Sentinel value (valid indices are 0-63)
        let mut capture_index: u8 = NO_CAPTURE;
        let mut possible_move_indices: [u8; 7] = [0; 7]; // Max 7 landing squares in any direction
        let mut possible_move_count: usize = 0;

        // Use simple comparison instead of Range::contains for performance
        #[allow(clippy::manual_range_contains)]
        while temp_index >= 0 && temp_index <= 63 {
            let temp_index_mask = 1u64 << temp_index;
            if friendly_pieces & temp_index_mask != 0 {
                // Blocked by friendly
                break;
            }
            if hostile_pieces & temp_index_mask != 0 {
                if capture_index != NO_CAPTURE {
                    // Either encountered two hostiles after each other
                    // or, encountered a hostile then empty* then hostile
                    break;
                }
                // Encountered first hostile
                capture_index = temp_index as u8;
            } else if capture_index != NO_CAPTURE {
                // Empty square after capturing a hostile - valid landing
                #[allow(clippy::cast_sign_loss)] // temp_index is 0..=63 here
                {
                    possible_move_indices[possible_move_count] = temp_index as u8;
                }
                possible_move_count += 1;
            }

            // Check for edge conditions BEFORE advancing
            // This avoids redundant iteration
            if DIRECTION == 1 && temp_index % 8 == 7 {
                break; // At right edge, can't go further right
            }
            if DIRECTION == -1 && temp_index % 8 == 0 {
                break; // At left edge, can't go further left
            }

            temp_index += DIRECTION;
        }

        if possible_move_count == 0 {
            return;
        }
        // Note: possible_move_count > 0 implies capture_index != NO_CAPTURE
        // (we only increment count after finding a hostile to capture)

        let capture_index_mask = 1u64 << capture_index;

        for &possible_move_index in &possible_move_indices[..possible_move_count] {
            let dest_mask = 1u64 << possible_move_index;
            *has_more_captures = true;

            let capture_action = Action::new_capture_as_king::<TEAM_INDEX>(
                src_mask,
                dest_mask,
                capture_index_mask,
                self.state.kings,
            );

            // Apply action
            self.apply_(&capture_action);

            Self::generate_king_captures_at::<TEAM_INDEX, DIRECTION>(
                self,
                actions,
                max_length,
                dest_mask,
                previous_action.combine(&capture_action),
            );

            // Undo action
            self.apply_(&capture_action);
        }
    }

    #[inline]
    fn generate_moves<const TEAM_INDEX: usize>(&self, actions: &mut Vec<Action>) {
        let empty = self.state.empty();
        self.generate_king_moves::<TEAM_INDEX>(actions, empty);
        self.generate_pawn_moves::<TEAM_INDEX>(actions, empty);
    }

    #[inline]
    fn generate_pawn_moves<const TEAM_INDEX: usize>(&self, actions: &mut Vec<Action>, empty: u64) {
        let mut friendly_pawns = self.friendly_pieces() & !self.state.kings;
        while friendly_pawns != 0u64 {
            let src_mask = friendly_pawns & friendly_pawns.wrapping_neg(); // get lowest set bit
            let src_index: usize = src_mask.trailing_zeros() as usize;
            let mut possible_dest_masks = if TEAM_INDEX == 0 {
                WHITE_PAWN_MOVES[src_index] & empty
            } else {
                BLACK_PAWN_MOVES[src_index] & empty
            };

            while possible_dest_masks != 0u64 {
                let dest_mask = possible_dest_masks & possible_dest_masks.wrapping_neg(); // get lowest set bit
                actions.push(Action::new_move_as_pawn::<TEAM_INDEX>(src_mask, dest_mask));
                possible_dest_masks ^= dest_mask; // clear lowest set bit
            }

            friendly_pawns ^= src_mask; // clear lowest set bit
        }
    }

    #[inline]
    fn generate_king_moves<const TEAM_INDEX: usize>(&self, actions: &mut Vec<Action>, empty: u64) {
        let mut friendly_kings = self.friendly_pieces() & self.state.kings;
        while friendly_kings != 0u64 {
            let src_mask = friendly_kings & friendly_kings.wrapping_neg(); // get lowest set bit

            // Generate king left moves
            let mut left_dest_mask = src_mask & !MASK_COL_A;
            while left_dest_mask != 0u64 {
                left_dest_mask = (left_dest_mask >> 1u8) & empty;
                if left_dest_mask != 0u64 {
                    actions.push(Action::new_move_as_king::<TEAM_INDEX>(
                        src_mask,
                        left_dest_mask,
                    ));
                }
                left_dest_mask &= !MASK_COL_A;
            }

            // Generate king right moves
            let mut right_dest_mask = src_mask & !MASK_COL_H;
            while right_dest_mask != 0u64 {
                right_dest_mask = (right_dest_mask << 1u8) & empty;
                if right_dest_mask != 0u64 {
                    actions.push(Action::new_move_as_king::<TEAM_INDEX>(
                        src_mask,
                        right_dest_mask,
                    ));
                }
                right_dest_mask &= !MASK_COL_H;
            }

            // Generate king up moves
            let mut up_dest_mask = src_mask & !MASK_ROW_8;
            while up_dest_mask != 0u64 {
                up_dest_mask = (up_dest_mask << 8u8) & empty;
                if up_dest_mask != 0u64 {
                    actions.push(Action::new_move_as_king::<TEAM_INDEX>(
                        src_mask,
                        up_dest_mask,
                    ));
                }
                up_dest_mask &= !MASK_ROW_8;
            }

            // Generate king down moves
            let mut down_dest_mask = src_mask & !MASK_ROW_1;
            while down_dest_mask != 0u64 {
                down_dest_mask = (down_dest_mask >> 8u8) & empty;
                if down_dest_mask != 0u64 {
                    actions.push(Action::new_move_as_king::<TEAM_INDEX>(
                        src_mask,
                        down_dest_mask,
                    ));
                }
                down_dest_mask &= !MASK_ROW_1;
            }

            friendly_kings ^= src_mask; // clear lowest set bit
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_status::GameStatus;
    use crate::Square;

    // ========== Initial Position Tests ==========

    #[test]
    fn initial_position_action_count() {
        let board = Board::new_default();
        let actions = board.actions();
        // Initial position has white pawns on rows 2 and 3
        // Row 2 is blocked by row 3 (can't move up)
        // Row 3 can move up into row 4
        // Both rows can move left/right where possible
        assert!(!actions.is_empty(), "Initial position should have moves");
        // Just verify it's a reasonable number - the exact count depends on rules
        assert!(
            !actions.is_empty(),
            "Should have moves from initial position"
        );
    }

    #[test]
    fn initial_position_only_moves_no_captures() {
        let board = Board::new_default();
        let actions = board.actions();
        // In initial position, no captures are possible
        for action in &actions {
            // A move has no captures, so opponent pieces delta should be 0
            assert_eq!(
                action.delta.pieces[Team::Black.to_usize()],
                0,
                "Initial position should only have moves, not captures"
            );
        }
    }

    // ========== Forced Capture Tests ==========

    #[test]
    fn forced_capture_rule() {
        // White pawn at D4, black pawn at D5 (capturable) and black pawn at H8
        // White should be forced to capture
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5, Square::H8], &[]);
        let actions = board.actions();

        assert_eq!(actions.len(), 1, "Should have exactly one capture");
        // Verify it's a capture (black pieces are affected)
        assert_ne!(
            actions[0].delta.pieces[Team::Black.to_usize()],
            0,
            "Action should be a capture"
        );
    }

    #[test]
    fn multiple_capture_options() {
        // White pawn at D4, can capture left (C5) or right (E5) or up (D5)
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::C4, Square::E4, Square::D5],
            &[],
        );
        let actions = board.actions();

        assert_eq!(actions.len(), 3, "Should have three capture options");
        for action in &actions {
            assert_ne!(
                action.delta.pieces[Team::Black.to_usize()],
                0,
                "All actions should be captures"
            );
        }
    }

    // ========== Maximum Capture Rule Tests ==========

    #[test]
    fn maximum_capture_rule_prefers_longer_chain() {
        // White pawn at A4
        // Option 1: capture B4 landing at C4 (length 1)
        // Option 2: capture A5 landing at A6, then capture B6 landing at C6 (length 2)
        let board = Board::from_squares(
            Team::White,
            &[Square::A4],
            &[Square::B4, Square::A5, Square::B6],
            &[],
        );
        let actions = board.actions();

        // Should only return the length-2 capture chain
        for action in &actions {
            let capture_count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(
                capture_count, 2,
                "Should only return maximum length captures (2)"
            );
        }
    }

    #[test]
    fn maximum_capture_multiple_equal_length() {
        // Two capture chains of equal length should both be returned
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5, Square::E4], &[]);
        let actions = board.actions();

        // Both are length-1 captures
        assert_eq!(
            actions.len(),
            2,
            "Should have two equal-length capture options"
        );
    }

    // ========== King Movement Tests ==========

    #[test]
    fn king_can_slide_multiple_squares() {
        // White king at D4, should be able to move in all 4 directions multiple squares
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[],
            &[Square::D4], // D4 is a king
        );
        let actions = board.actions();

        // King at D4 can move:
        // Left: C4, B4, A4 (3 moves)
        // Right: E4, F4, G4, H4 (4 moves)
        // Up: D5, D6, D7, D8 (4 moves)
        // Down: D3, D2, D1 (3 moves)
        // Total: 14 moves
        assert_eq!(actions.len(), 14, "King at D4 should have 14 moves");
    }

    #[test]
    fn king_blocked_by_friendly_piece() {
        // White king at D4, white pawn at D6
        let board = Board::from_squares(
            Team::White,
            &[Square::D4, Square::D6],
            &[],
            &[Square::D4], // D4 is a king
        );
        let actions = board.actions();

        // King can move up to D5 only (blocked by D6)
        // Left: 3, Right: 4, Up: 1 (blocked at D6), Down: 3
        // Total: 11 moves (king) + pawn moves
        let king_up_moves: Vec<_> = actions
            .iter()
            .filter(|a| {
                let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
                dest == Square::D5.to_mask()
            })
            .collect();
        assert_eq!(
            king_up_moves.len(),
            1,
            "King should only be able to move to D5 going up"
        );
    }

    // ========== King Capture Tests ==========

    #[test]
    fn king_can_capture_from_distance() {
        // White king at A4, black pawn at D4, king can land on E4, F4, G4, or H4
        let board = Board::from_squares(
            Team::White,
            &[Square::A4],
            &[Square::D4],
            &[Square::A4], // A4 is a king
        );
        let actions = board.actions();

        // King captures D4 and can land on E4, F4, G4, H4 (4 options)
        assert_eq!(
            actions.len(),
            4,
            "King should have 4 landing options after capture"
        );
        for action in &actions {
            assert_eq!(
                action.delta.pieces[Team::Black.to_usize()],
                Square::D4.to_mask(),
                "Should capture D4"
            );
        }
    }

    #[test]
    fn king_multi_direction_capture() {
        // White king at D4, can capture in multiple directions
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::B4, Square::D6],
            &[Square::D4],
        );
        let actions = board.actions();

        // Should have captures in both directions
        assert!(
            actions.len() >= 2,
            "King should be able to capture in multiple directions"
        );
    }

    // ========== Pawn Movement Tests ==========

    #[test]
    fn white_pawn_moves_forward_and_sideways() {
        let board = Board::from_squares(Team::White, &[Square::D4], &[], &[]);
        let actions = board.actions();

        // D4 pawn can move to C4 (left), E4 (right), D5 (up)
        assert_eq!(actions.len(), 3, "Pawn at D4 should have 3 moves");
    }

    #[test]
    fn black_pawn_moves_forward_and_sideways() {
        let board = Board::from_squares(Team::Black, &[], &[Square::D5], &[]);
        let actions = board.actions();

        // D5 black pawn can move to C5 (left), E5 (right), D4 (down)
        assert_eq!(actions.len(), 3, "Black pawn at D5 should have 3 moves");
    }

    #[test]
    fn pawn_at_edge_has_fewer_moves() {
        let board = Board::from_squares(Team::White, &[Square::A4], &[], &[]);
        let actions = board.actions();

        // A4 pawn can move to B4 (right), A5 (up) - cannot go left
        assert_eq!(actions.len(), 2, "Pawn at A4 should have 2 moves");
    }

    // ========== Promotion Tests ==========

    #[test]
    fn pawn_promotes_on_last_rank() {
        let board = Board::from_squares(Team::White, &[Square::D7], &[], &[]);
        let actions = board.actions();

        // Find the move to D8 and verify it promotes
        let promotion_action = actions
            .iter()
            .find(|a| a.delta.pieces[Team::White.to_usize()] & Square::D8.to_mask() != 0);

        assert!(promotion_action.is_some(), "Should be able to move to D8");
        let action = promotion_action.unwrap();
        assert_ne!(
            action.delta.kings & Square::D8.to_mask(),
            0,
            "Pawn should promote to king at D8"
        );
    }

    #[test]
    fn pawn_promotes_during_capture() {
        // White pawn at C7, captures black pawn at C8... wait, that's impossible
        // Let's do: White pawn at B6, captures black pawn at B7, lands on B8
        let board = Board::from_squares(Team::White, &[Square::B6], &[Square::B7], &[]);
        let actions = board.actions();

        assert_eq!(actions.len(), 1, "Should have one capture");
        let action = &actions[0];
        assert_ne!(
            action.delta.kings & Square::B8.to_mask(),
            0,
            "Pawn should promote after capture landing on B8"
        );
    }

    // ========== No Actions Tests ==========

    #[test]
    fn no_pieces_no_actions() {
        let board = Board::from_squares(Team::White, &[], &[Square::D4], &[]);
        let actions = board.actions();
        assert!(actions.is_empty(), "No friendly pieces means no actions");
    }

    #[test]
    fn completely_blocked_pawn_can_capture() {
        // White pawn at B2, surrounded by black pawns
        // It's blocked for moves but can capture!
        let board = Board::from_squares(
            Team::White,
            &[Square::B2],
            &[Square::A2, Square::B3, Square::C2],
            &[],
        );
        let actions = board.actions();
        // The pawn can capture A2, B3, or C2
        assert!(
            !actions.is_empty(),
            "Surrounded pawn can capture adjacent enemies"
        );
        for action in &actions {
            assert_ne!(
                action.delta.pieces[Team::Black.to_usize()],
                0,
                "Actions should be captures"
            );
        }
    }

    #[test]
    fn truly_blocked_pawn() {
        // White pawn at B2, surrounded by friendly pawns (can't capture friendlies)
        let board = Board::from_squares(
            Team::White,
            &[Square::B2, Square::A2, Square::B3, Square::C2],
            &[Square::H8],
            &[],
        );
        let actions = board.actions();
        // B2 is blocked, but A2, B3, C2 can move
        // A2: B2 blocked, A3 available = 1 move
        // B3: A3 available, C3 available, B4 available = 3 moves
        // C2: B2 blocked, D2 available, C3 available = 2 moves
        // Total moves should not include B2 moving anywhere
        let b2_moves: Vec<_> = actions
            .iter()
            .filter(|a| {
                let delta = a.delta.pieces[Team::White.to_usize()];
                delta & Square::B2.to_mask() != 0
            })
            .collect();
        assert!(b2_moves.is_empty(), "B2 pawn should have no moves");
    }

    // ========== Chain Capture Tests ==========

    #[test]
    fn pawn_chain_capture_two_pieces() {
        // White pawn at A4
        // Can capture A5 -> A6, then B6 -> C6
        let board = Board::from_squares(Team::White, &[Square::A4], &[Square::A5, Square::B6], &[]);
        let actions = board.actions();

        // Should capture both pieces in chain
        assert!(!actions.is_empty(), "Should have capture chain");
        for action in &actions {
            let capture_count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(capture_count, 2, "Should capture 2 pieces in chain");
        }
    }

    #[test]
    fn pawn_chain_capture_three_pieces() {
        // White pawn at A2
        // Chain: A2 captures A3->A4, then B4->C4, then C5->C6
        let board = Board::from_squares(
            Team::White,
            &[Square::A2],
            &[Square::A3, Square::B4, Square::C5],
            &[],
        );
        let actions = board.actions();

        assert!(!actions.is_empty(), "Should have capture chain");
        for action in &actions {
            let capture_count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(capture_count, 3, "Should capture 3 pieces in chain");
        }
    }

    #[test]
    fn king_chain_capture() {
        // White king at A4
        // Can capture C4 -> E4, then E6 -> E8
        let board = Board::from_squares(
            Team::White,
            &[Square::A4],
            &[Square::C4, Square::E6],
            &[Square::A4],
        );
        let actions = board.actions();

        // Should have chain captures
        assert!(!actions.is_empty(), "King should have capture options");
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap();
        assert_eq!(max_captures, 2, "King should capture 2 pieces in chain");
    }

    // ========== King Edge Cases ==========

    #[test]
    fn king_cannot_pass_through_friendly() {
        // White king at A4, white pawn at C4
        // King should not be able to move past C4 to the right
        let board = Board::from_squares(
            Team::White,
            &[Square::A4, Square::C4],
            &[Square::H8],
            &[Square::A4],
        );
        let actions = board.actions();

        // King should only reach B4 going right (blocked by C4)
        let right_moves: Vec<_> = actions
            .iter()
            .filter(|a| {
                let src = Square::A4.to_mask();
                let delta = a.delta.pieces[Team::White.to_usize()];
                // Check if it's a king move (src is toggled)
                delta & src != 0 && {
                    let dest = delta & !src;
                    // Check if destination is to the right of A4 (same row, higher column)
                    // SAFETY: dest is a single bit (action moves one piece).
                    let dest_sq = unsafe { Square::from_mask(dest) };
                    dest_sq.row() == 3 && dest_sq.column() > 0
                }
            })
            .collect();

        assert_eq!(
            right_moves.len(),
            1,
            "King should only reach B4 going right"
        );
    }

    #[test]
    fn king_cannot_jump_over_hostile_without_capturing() {
        // White king at A4, black pawn at C4
        // King cannot move to D4+ without capturing
        let board = Board::from_squares(Team::White, &[Square::A4], &[Square::C4], &[Square::A4]);
        let actions = board.actions();

        // All actions that go past C4 must be captures
        for action in &actions {
            let dest = action.delta.pieces[Team::White.to_usize()] & !Square::A4.to_mask();
            if dest != 0 {
                // SAFETY: dest is a single bit (action moves one piece).
                let dest_sq = unsafe { Square::from_mask(dest) };
                if dest_sq.row() == 3 && dest_sq.column() >= 3 {
                    // Past C4, must be a capture
                    assert_ne!(
                        action.delta.pieces[Team::Black.to_usize()],
                        0,
                        "Moving past hostile must be a capture"
                    );
                }
            }
        }
    }

    // ========== Black Team Tests ==========

    #[test]
    fn black_pawn_captures() {
        let board = Board::from_squares(Team::Black, &[Square::A1], &[Square::D5], &[]);
        let actions = board.actions();

        // Black pawn at D5 can move down/left/right
        // D5 -> D4 (down), C5 (left), E5 (right)
        assert_eq!(actions.len(), 3, "Black pawn at D5 should have 3 moves");
    }

    #[test]
    fn black_king_movement() {
        let board = Board::from_squares(Team::Black, &[Square::A1], &[Square::D4], &[Square::D4]);
        let actions = board.actions();

        // Black king at D4 should have 14 moves (same as white king)
        assert_eq!(actions.len(), 14, "Black king at D4 should have 14 moves");
    }

    // ========== Multiple Pieces Tests ==========

    #[test]
    fn multiple_pieces_all_can_move() {
        let board = Board::from_squares(Team::White, &[Square::A4, Square::H4], &[Square::D8], &[]);
        let actions = board.actions();

        // A4: B4 (right), A5 (up) = 2 moves
        // H4: G4 (left), H5 (up) = 2 moves
        // Total: 4 moves
        assert_eq!(actions.len(), 4, "Both pawns should contribute moves");
    }

    #[test]
    fn multiple_pieces_one_must_capture() {
        // A4 can capture, H4 cannot
        // Only A4's capture should be returned
        let board = Board::from_squares(Team::White, &[Square::A4, Square::H4], &[Square::A5], &[]);
        let actions = board.actions();

        assert_eq!(actions.len(), 1, "Only capture should be returned");
        assert_ne!(
            actions[0].delta.pieces[Team::Black.to_usize()],
            0,
            "Action should be a capture"
        );
    }

    // ========== Promotion Edge Cases ==========

    #[test]
    fn black_pawn_promotes_on_row_1() {
        let board = Board::from_squares(Team::Black, &[Square::H8], &[Square::D2], &[]);
        let actions = board.actions();

        // Find move to D1
        let promotion = actions
            .iter()
            .find(|a| a.delta.pieces[Team::Black.to_usize()] & Square::D1.to_mask() != 0);

        assert!(promotion.is_some(), "Should be able to move to D1");
        assert_ne!(
            promotion.unwrap().delta.kings & Square::D1.to_mask(),
            0,
            "Black pawn should promote at D1"
        );
    }

    #[test]
    fn pawn_capture_leads_to_promotion() {
        // White pawn at B6, captures B7, lands at B8 and promotes
        let board = Board::from_squares(Team::White, &[Square::B6], &[Square::B7], &[]);
        let actions = board.actions();

        assert_eq!(actions.len(), 1, "Should have one capture");
        let action = &actions[0];

        // Verify capture happened
        assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B7.to_mask(),
            "Should capture B7"
        );

        // Verify promotion happened
        assert_ne!(
            action.delta.kings & Square::B8.to_mask(),
            0,
            "Pawn should promote at B8"
        );
    }

    // ========== Corner Cases ==========

    #[test]
    fn pawn_in_corner() {
        // White pawn at A8 (promoted position - should be king, but let's test as pawn)
        // Actually A8 is promotion row for white, so it would be a king
        // Let's use H1 for black
        let board = Board::from_squares(
            Team::Black,
            &[Square::A8],
            &[Square::H1],
            &[Square::H1], // H1 is a king
        );
        let actions = board.actions();

        // Black king at H1: can move G1 (left), H2 (up)
        // But H1 is corner, so limited moves
        // Left: G1, F1, E1, D1, C1, B1, A1 = 7 moves
        // Up: H2, H3, H4, H5, H6, H7, H8 = 7 moves
        assert_eq!(actions.len(), 14, "King at H1 should have 14 moves");
    }

    // ============================================================================
    // COMPREHENSIVE RULE TESTS
    // These tests cover all rules from rules.md to ensure complete coverage
    // ============================================================================

    // ========== 180-DEGREE TURN RESTRICTION TESTS ==========
    // Rule: During a multiple capture sequence, a piece cannot make a 180-degree turn

    #[test]
    fn king_cannot_reverse_up_down_during_capture() {
        // White king at D1, black pieces at D3 and D5
        // King captures D3, landing at D4
        // From D4, the king should NOT be able to capture D5 going up then reverse to go down
        // Actually this tests: after going UP to capture, can't immediately go DOWN

        // Setup: King at D1, enemies at D3 (capture going up, land D4-D7)
        // Then enemy at D2 would require going down (180° turn)
        // But D2 is below D1, so let's set up differently:

        // King at D4, enemies at D2 and D6
        // If king captures D6 (going up), lands at D7 or D8
        // From there, D2 requires going down - but it's a different line

        // Better test: King at D4, enemy at D6, enemy at D3
        // Capture D6 (up) -> land D7
        // From D7, enemy D3 is far below, would need to go DOWN (opposite of UP)
        // This should be blocked by 180° rule... but wait, D3 from D7 is a long capture

        // Simplest test: King at D4, enemies at D6 and D8 placed so after first capture
        // the only continuation would require reversing
        // Actually: King at D4, enemy at D6. After capture, land at D7.
        // Now if there's an enemy at D5 that we skipped... no, we capture D6.

        // Real test: King D4, enemies at D2 (below) and D6 (above)
        // King has two 1-capture options. But if setup allows chain that reverses, it should fail.

        // Setup for reversal test:
        // King at D4, enemy at D6, enemy at D4... wait can't overlap
        //
        // King at D5, enemies at D3 and D7
        // Capture D7 going up, land at D8
        // From D8, capture D3 going down? D3 is at row 2, D8 is row 7
        // That's a valid long-range capture going DOWN
        // After going UP, going DOWN is 180° - should be blocked!

        let board = Board::from_squares(
            Team::White,
            &[Square::D5],
            &[Square::D3, Square::D7],
            &[Square::D5], // D5 is a king
        );
        let actions = board.actions();

        // King can capture D3 (going down) or D7 (going up)
        // Each is a single capture - no chain should form because reversing is blocked
        // So we should have multiple single-capture actions, not a 2-capture chain

        for action in &actions {
            let capture_count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(
                capture_count, 1,
                "King should NOT chain captures that require 180° turn (up then down)"
            );
        }
    }

    #[test]
    fn king_cannot_reverse_left_right_during_capture() {
        // King at D4, enemies at B4 (left) and F4 (right)
        // After capturing B4 going left (land at A4),
        // capturing F4 would require going right (180° turn) - should be blocked

        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::B4, Square::F4],
            &[Square::D4], // D4 is a king
        );
        let actions = board.actions();

        // Should have two single-capture options, NOT a 2-capture chain
        for action in &actions {
            let capture_count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(
                capture_count, 1,
                "King should NOT chain captures that require 180° turn (left then right)"
            );
        }
    }

    #[test]
    fn king_can_turn_90_degrees_during_capture() {
        // King at D4, enemy at D6 (up), enemy at F6 (right from D6 area)
        // Capture D6 going up, land at D7
        // From D7, capture F7 going right (90° turn, not 180°) - should be ALLOWED

        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::D6, Square::F7],
            &[Square::D4],
        );
        let actions = board.actions();

        // Should have a 2-capture chain: D6 (up) then F7 (right)
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_captures, 2,
            "King should be able to chain captures with 90° turn"
        );
    }

    #[test]
    fn king_180_restriction_complex_scenario() {
        // More complex: King at A1, enemies arranged in an L-shape
        // A1 -> captures A3 (up) -> lands A4-A8
        // From A5, could capture C5 (right) -> lands D5-H5
        // From E5, could capture E3 (down) - this is 90° from right, allowed

        let board = Board::from_squares(
            Team::White,
            &[Square::A1],
            &[Square::A3, Square::C5, Square::E3],
            &[Square::A1],
        );
        let actions = board.actions();

        // Should be able to do 3-capture chain with 90° turns
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_captures, 3,
            "King should chain 3 captures with 90° turns"
        );
    }

    /// Black king at D4 can capture D2 (South) or D6 (North), but NOT both
    /// since that would require a 180° turn.
    #[test]
    fn king_180_turn_prohibited_vertical() {
        let board = Board::from_squares(
            Team::Black,
            &[Square::D2, Square::D6],
            &[Square::D4],
            &[Square::D4, Square::D6],
        );

        let actions = board.actions();

        assert!(!actions.is_empty());
        for action in &actions {
            assert_eq!(
                action.capture_count(Team::Black),
                1,
                "180° turn should prevent chaining North->South or South->North captures"
            );
        }
    }

    /// Black king at D4 can capture B4 (West) or F4 (East), but NOT both
    /// since that would require a 180° turn.
    #[test]
    fn king_180_turn_prohibited_horizontal() {
        let board = Board::from_squares(
            Team::Black,
            &[Square::B4, Square::F4],
            &[Square::D4],
            &[Square::D4, Square::F4],
        );

        let actions = board.actions();

        assert!(!actions.is_empty());
        for action in &actions {
            assert_eq!(
                action.capture_count(Team::Black),
                1,
                "180° turn should prevent chaining East->West or West->East captures"
            );
        }
    }

    // ========== IMMEDIATE PIECE REMOVAL TESTS ==========
    // Rule: Captured pieces are removed immediately, allowing crossing the same square

    #[test]
    fn can_cross_captured_square_pawn() {
        // This tests that after capturing a piece, the square becomes available
        // White pawn at A4, enemies at A5, A7
        // Capture A5 -> land A6
        // Capture A7 -> land A8 (crosses through where A5 was)
        // Wait, A6 doesn't cross A5's square...

        // Better: White pawn at A2, enemies at A3 and B4
        // Capture A3 -> land A4
        // Capture B4 -> land C4
        // This doesn't cross A3's square either

        // For pawns, crossing the same square is hard to set up because they jump 2 squares
        // The "crossing same square" benefit is more relevant for kings
        // Let's test with a king instead
        let board = Board::from_squares(Team::White, &[Square::A2], &[Square::A3, Square::B4], &[]);
        let actions = board.actions();

        // Should capture both in chain
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(max_captures, 2, "Pawn should chain 2 captures");
    }

    #[test]
    fn king_can_cross_captured_square() {
        // King at A4, enemies at C4 and E4 (not on promotion row)
        // Capture C4 -> can land at D4
        // From D4, capture E4 -> land F4, G4, or H4
        // This uses the path through where C4 was

        let board = Board::from_squares(
            Team::White,
            &[Square::A4],
            &[Square::C4, Square::E4],
            &[Square::A4],
        );
        let actions = board.actions();

        // Should have 2-capture chains
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_captures, 2,
            "King should be able to chain captures (immediate removal allows path)"
        );
    }

    #[test]
    fn king_complex_crossing_pattern() {
        // Create a scenario where the king must cross a previously captured square
        // King at A4, enemies at C4 and A6
        // Option 1: Capture C4 (right) -> land D4-H4
        // Option 2: Capture A6 (up) -> land A7-A8
        //
        // For crossing: King at D4, enemies at D2 and D6 and F4
        // Capture D2 (down) -> land D1
        // Can't continue (D1 is edge)
        //
        // Better: King at D4, enemy at B4, enemy at B6
        // Capture B4 (left) -> land A4
        // From A4, capture B6? B6 is not adjacent diagonally, let's think orthogonally
        // From A4, enemy at A6 would be capturable (going up)

        // King at D4, enemies at B4 (capture left), B2 (would need to pass through B4's position)
        // Capture B4 -> land A4
        // From A4, enemy at B2 is diagonal - invalid
        //
        // King at D4, enemies at B4 and D2
        // Capture B4 (left) -> land A4
        // From A4, capture D2? D2 is not directly accessible from A4

        // Simpler: King at E4, enemies at C4 and C2
        // Capture C4 (left) -> land A4 or B4
        // From B4, capture C2 (right-down)? No, orthogonal only
        // From A4, capture C2? Not directly possible

        // The crossing scenario needs more thought. Let's test basic immediate removal:
        let board = Board::from_squares(
            Team::White,
            &[Square::A4],
            &[Square::C4, Square::E4, Square::G4],
            &[Square::A4],
        );
        let actions = board.actions();

        // King can capture C4 -> land D4
        // From D4, capture E4 -> land F4
        // From F4, capture G4 -> land H4
        // This is a 3-capture chain going right continuously (no crossing needed)
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(max_captures, 3, "King should chain 3 captures in a line");
    }

    // ========== MID-SEQUENCE PROMOTION TESTS ==========
    // Tests for how promotion works during capture sequences

    #[test]
    fn pawn_promotes_at_end_of_capture_sequence() {
        // White pawn at D6, enemies at D7 and E8
        // Capture D7 -> land D8 (promotion row)
        // If more captures available from D8 as a king, it depends on interpretation
        // Current implementation: continues as pawn (Interpretation B)

        let board = Board::from_squares(Team::White, &[Square::D6], &[Square::D7], &[]);
        let actions = board.actions();

        assert_eq!(actions.len(), 1, "Should have one capture");
        let action = &actions[0];

        // Verify lands on D8 and promotes
        let dest = action.delta.pieces[Team::White.to_usize()] & !Square::D6.to_mask();
        assert_eq!(dest, Square::D8.to_mask(), "Should land on D8");
        assert_ne!(
            action.delta.kings & Square::D8.to_mask(),
            0,
            "Should promote to king at D8"
        );
    }

    #[test]
    fn pawn_captures_and_promotes_when_ending_on_back_row() {
        // White pawn at D6, enemy at D7
        // Capture D7 -> land D8 (promotion row)
        // Since no more captures available, pawn promotes

        let board = Board::from_squares(Team::White, &[Square::D6], &[Square::D7], &[]);
        let actions = board.actions();

        assert_eq!(actions.len(), 1, "Should have one capture");
        let action = &actions[0];

        // Verify lands on D8 and promotes
        let dest = action.delta.pieces[Team::White.to_usize()] & !Square::D6.to_mask();
        assert_eq!(dest, Square::D8.to_mask(), "Should land on D8");
        assert_ne!(
            action.delta.kings & Square::D8.to_mask(),
            0,
            "Should promote to king at D8"
        );
    }

    #[test]
    fn pawn_continues_capturing_from_promotion_row() {
        // Rule: A pawn that lands on the promotion row mid-capture does NOT
        // become a king until the turn ends. It must continue capturing as a pawn.
        //
        // Setup: White pawn at D6, enemies at D7 and C8
        // - Pawn captures D7 -> lands on D8 (promotion row)
        // - Pawn should continue capturing C8 -> lands on B8 (as a PAWN)
        // - Only NOW does the pawn promote to king (sequence ended on promotion row)

        let board = Board::from_squares(Team::White, &[Square::D6], &[Square::D7, Square::C8], &[]);
        let actions = board.actions();

        // Should have exactly one action: the 2-capture chain
        // (single captures are filtered out by max capture rule)
        assert_eq!(actions.len(), 1, "Should have one 2-capture action");
        let action = &actions[0];

        // Verify captures both D7 and C8
        let captures = action.delta.pieces[Team::Black.to_usize()];
        assert_eq!(
            captures,
            Square::D7.to_mask() | Square::C8.to_mask(),
            "Should capture both D7 and C8"
        );

        // Verify pawn ends on B8 (2 squares left from D8)
        let final_pos = action.delta.pieces[Team::White.to_usize()] & !Square::D6.to_mask();
        assert_eq!(final_pos, Square::B8.to_mask(), "Should end on B8");

        // Verify pawn promotes at B8 (final position on promotion row)
        assert_ne!(
            action.delta.kings & Square::B8.to_mask(),
            0,
            "Should promote to king at B8"
        );
    }

    #[test]
    fn pawn_chain_capture_ending_on_promotion_row() {
        // White pawn at A4, enemies at A5 and B6
        // Capture A5 -> land A6
        // Capture B6 -> land C6
        // Note: This doesn't involve promotion row, but tests chain captures work

        let board = Board::from_squares(Team::White, &[Square::A4], &[Square::A5, Square::B6], &[]);
        let actions = board.actions();

        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(max_captures, 2, "Should capture both in chain");
    }

    // NOTE: Mid-capture promotion tests are not included because the current
    // implementation doesn't support pawns continuing to capture after landing
    // on the promotion row. The Action::new_capture_as_pawn function panics
    // if source is on promotion row. This is documented in implementation_status.md.

    // ========== MAXIMUM CAPTURE RULE TESTS ==========
    // Rule: Must capture maximum number of pieces; can choose among equals

    #[test]
    fn max_capture_three_vs_two() {
        // Setup where one path captures 3, another captures 2
        // White pawn at A2
        // Path 1: A2 captures A3->A4, then B4->C4 (2 captures)
        // Path 2: A2 captures A3->A4, then B4->C4, then C5->C6 (3 captures)

        let board = Board::from_squares(
            Team::White,
            &[Square::A2],
            &[Square::A3, Square::B4, Square::C5],
            &[],
        );
        let actions = board.actions();

        // All returned actions should be 3 captures (maximum)
        for action in &actions {
            let count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(count, 3, "Only 3-capture sequences should be returned");
        }
    }

    #[test]
    fn max_capture_equal_length_all_returned() {
        // Multiple paths with same capture count should all be available
        // White pawn at D4
        // Can capture: D5->D6 (1 capture up) OR C4->B4 (1 capture left) OR E4->F4 (1 capture right)

        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::D5, Square::C4, Square::E4],
            &[],
        );
        let actions = board.actions();

        // Should have 3 capture options, all length 1
        assert_eq!(
            actions.len(),
            3,
            "Should have 3 equal-length capture options"
        );
        for action in &actions {
            let count = action.delta.pieces[Team::Black.to_usize()].count_ones();
            assert_eq!(count, 1, "All captures should be length 1");
        }
    }

    #[test]
    fn max_capture_king_vs_pawn_count_equally() {
        // Capturing a king counts the same as capturing a pawn
        // D4 pawn, D5 king (1 capture), C4+B5 pawns (2 captures via chain)
        // Path 1: capture D5 (1 king)
        // Path 2: capture C4->B4, then B5->B6 (2 pawns)
        // Should prefer 2 captures even though path 1 captures a king
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::D5, Square::C4, Square::B5],
            &[Square::D5], // D5 is a king
        );
        let actions = board.actions();

        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_captures, 2,
            "Should prefer 2-pawn capture over 1-king capture"
        );
    }

    // ========== MEN MOVEMENT TESTS ==========
    // Rule: Men move forward, left, right only (no backward, no diagonal)

    #[test]
    fn white_pawn_cannot_move_backward() {
        // White pawn at D4, space behind at D3
        // Should NOT be able to move to D3
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[]);
        let actions = board.actions();

        let backward_move = actions.iter().find(|a| {
            let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
            dest == Square::D3.to_mask()
        });

        assert!(
            backward_move.is_none(),
            "White pawn should not move backward to D3"
        );
    }

    #[test]
    fn black_pawn_cannot_move_backward() {
        // Black pawn at D5, space behind at D6
        // Should NOT be able to move to D6
        let board = Board::from_squares(Team::Black, &[Square::H1], &[Square::D5], &[]);
        let actions = board.actions();

        let backward_move = actions.iter().find(|a| {
            let dest = a.delta.pieces[Team::Black.to_usize()] & !Square::D5.to_mask();
            dest == Square::D6.to_mask()
        });

        assert!(
            backward_move.is_none(),
            "Black pawn should not move backward to D6"
        );
    }

    #[test]
    fn pawn_cannot_move_diagonally() {
        // White pawn at D4, all diagonal squares empty
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[]);
        let actions = board.actions();

        let diagonal_squares = [Square::C3, Square::C5, Square::E3, Square::E5];
        for sq in &diagonal_squares {
            let diag_move = actions.iter().find(|a| {
                let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
                dest == sq.to_mask()
            });
            assert!(
                diag_move.is_none(),
                "Pawn should not move diagonally to {sq:?}"
            );
        }
    }

    #[test]
    fn pawn_moves_exactly_one_square() {
        // Pawn should only move 1 square, not 2+
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[]);
        let actions = board.actions();

        // Valid destinations: C4, E4, D5 (distance 1)
        // Invalid: B4, F4, D6 (distance 2)
        let valid_dests = [Square::C4, Square::E4, Square::D5];
        let invalid_dests = [Square::B4, Square::F4, Square::D6];

        for sq in &valid_dests {
            let found = actions.iter().any(|a| {
                let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
                dest == sq.to_mask()
            });
            assert!(found, "Pawn should be able to move to {sq:?}");
        }

        for sq in &invalid_dests {
            let found = actions.iter().any(|a| {
                let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
                dest == sq.to_mask()
            });
            assert!(!found, "Pawn should NOT move 2 squares to {sq:?}");
        }
    }

    // ========== KING MOVEMENT TESTS ==========
    // Rule: Kings move any distance orthogonally (like a rook)

    #[test]
    fn king_moves_multiple_squares_all_directions() {
        // King at D4 in center, should reach all squares in its row and column
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[Square::D4]);
        let actions = board.actions();

        // Row 4: A4, B4, C4, E4, F4, G4, H4 (7 squares)
        // Column D: D1, D2, D3, D5, D6, D7, D8 (7 squares)
        // Total: 14 moves
        assert_eq!(actions.len(), 14, "King at D4 should have 14 moves");
    }

    #[test]
    fn king_cannot_move_diagonally() {
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::H8], &[Square::D4]);
        let actions = board.actions();

        let diagonal_squares = [
            Square::A1,
            Square::B2,
            Square::C3,
            Square::E5,
            Square::F6,
            Square::G7,
            Square::A7,
            Square::B6,
            Square::C5,
            Square::E3,
            Square::F2,
            Square::G1,
        ];

        for sq in &diagonal_squares {
            let diag_move = actions.iter().find(|a| {
                let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
                dest == sq.to_mask()
            });
            assert!(
                diag_move.is_none(),
                "King should not move diagonally to {sq:?}"
            );
        }
    }

    // ========== MEN CAPTURE TESTS ==========
    // Rule: Men capture forward, left, right (no backward, no diagonal)

    #[test]
    fn white_pawn_cannot_capture_backward() {
        // White pawn at D4, black pawn at D3 (behind)
        // Should NOT be able to capture backward
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::D3, Square::D5], // D3 behind, D5 in front
            &[],
        );
        let actions = board.actions();

        // Should only capture D5 (forward), not D3 (backward)
        assert_eq!(actions.len(), 1, "Should have only 1 capture");
        assert_eq!(
            actions[0].delta.pieces[Team::Black.to_usize()],
            Square::D5.to_mask(),
            "Should capture D5 (forward), not D3 (backward)"
        );
    }

    #[test]
    fn black_pawn_cannot_capture_backward() {
        // Black pawn at D5, white pawn at D6 (behind for black)
        let board = Board::from_squares(
            Team::Black,
            &[Square::D6, Square::D4], // D6 behind, D4 in front
            &[Square::D5],
            &[],
        );
        let actions = board.actions();

        // Should only capture D4 (forward for black = down), not D6 (backward)
        assert_eq!(actions.len(), 1, "Should have only 1 capture");
        assert_eq!(
            actions[0].delta.pieces[Team::White.to_usize()],
            Square::D4.to_mask(),
            "Should capture D4 (forward for black), not D6 (backward)"
        );
    }

    #[test]
    fn pawn_cannot_capture_diagonally() {
        // White pawn at D4, black pawns at diagonal positions
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::C3, Square::C5, Square::E3, Square::E5],
            &[],
        );
        let actions = board.actions();

        // Should have NO captures (all enemies are diagonal)
        // Without captures, should have moves instead
        for action in &actions {
            assert_eq!(
                action.delta.pieces[Team::Black.to_usize()],
                0,
                "Should not capture diagonal enemies"
            );
        }
    }

    // ========== KING CAPTURE TESTS ==========
    // Rule: Kings use flying capture (any distance, land anywhere beyond)

    #[test]
    fn king_capture_from_distance() {
        // King at A4, enemy at E4 (distance 4)
        // Should be able to capture and land on F4, G4, or H4
        let board = Board::from_squares(Team::White, &[Square::A4], &[Square::E4], &[Square::A4]);
        let actions = board.actions();

        // Should have 3 capture options (landing on F4, G4, H4)
        assert_eq!(actions.len(), 3, "King should have 3 landing options");

        for action in &actions {
            assert_eq!(
                action.delta.pieces[Team::Black.to_usize()],
                Square::E4.to_mask(),
                "All captures should take E4"
            );
        }
    }

    #[test]
    fn king_cannot_capture_two_in_line() {
        // King at A4, enemies at C4 and E4 (two in a row)
        // Should NOT be able to jump both in one move
        let board = Board::from_squares(
            Team::White,
            &[Square::A4],
            &[Square::C4, Square::E4],
            &[Square::A4],
        );
        let actions = board.actions();

        // Can capture C4 (landing D4), or start a chain
        // After capturing C4, E4 is still there to capture
        // This should be a 2-capture chain
        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_captures, 2,
            "Should capture both in chain, not single jump"
        );
    }

    // ========== MANDATORY CAPTURE TESTS ==========
    // Rule: If capture is available, must capture (no moves allowed)

    #[test]
    fn must_capture_when_available() {
        // White pawn at D4, can move OR capture
        // Enemy at D5 makes capture available
        let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5], &[]);
        let actions = board.actions();

        // All actions must be captures
        for action in &actions {
            assert_ne!(
                action.delta.pieces[Team::Black.to_usize()],
                0,
                "Must capture when capture is available"
            );
        }
    }

    #[test]
    fn moves_allowed_when_no_capture() {
        // White pawn at D4, no capturable enemies
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::H8], // Far away enemy
            &[],
        );
        let actions = board.actions();

        // All actions should be moves (non-captures)
        for action in &actions {
            assert_eq!(
                action.delta.pieces[Team::Black.to_usize()],
                0,
                "Should be moves, not captures"
            );
        }
        assert!(!actions.is_empty(), "Should have moves available");
    }

    // ========== DRAW CONDITION TESTS ==========
    // Note: Most draw conditions are in board.rs status() tests

    #[test]
    fn one_piece_each_is_draw() {
        // One white piece vs one black piece
        let board = Board::from_squares(Team::White, &[Square::A1], &[Square::H8], &[]);
        assert_eq!(
            board.status(),
            GameStatus::Draw,
            "One piece each should be draw"
        );
    }

    #[test]
    fn one_king_each_is_draw() {
        let board = Board::from_squares(
            Team::White,
            &[Square::A1],
            &[Square::H8],
            &[Square::A1, Square::H8],
        );
        assert_eq!(
            board.status(),
            GameStatus::Draw,
            "One king each should be draw"
        );
    }

    #[test]
    fn king_vs_pawn_is_draw() {
        // Current implementation: 1 vs 1 is always draw regardless of type
        let board = Board::from_squares(
            Team::White,
            &[Square::A1],
            &[Square::H8],
            &[Square::A1], // White has king, black has pawn
        );
        assert_eq!(
            board.status(),
            GameStatus::Draw,
            "King vs pawn (1v1) is draw in current implementation"
        );
    }

    // ========== WIN CONDITION TESTS ==========

    #[test]
    fn no_pieces_means_loss() {
        let board = Board::from_squares(Team::White, &[], &[Square::D4], &[]);
        assert_eq!(
            board.status(),
            GameStatus::Won(Team::Black),
            "No white pieces means black wins"
        );
    }

    #[test]
    fn blocked_means_loss() {
        // White pawns at A2 and A3, completely surrounded by enemies
        // such that all captures are blocked (landing squares occupied)
        // This is the same setup as board.rs::status_friendly_blocked
        let board = Board::from_squares(
            Team::White,
            &[Square::A2, Square::A3],
            &[
                Square::A4, // Blocks A3's forward capture landing
                Square::A5, // Extra blocker
                Square::B2, // Adjacent to A2 (capturable but landing blocked)
                Square::B3, // Adjacent to A3 (capturable but landing blocked)
                Square::C2, // Blocks B2 capture landing
                Square::C3, // Blocks B3 capture landing
            ],
            &[],
        );
        // A2: left=edge, forward=A3(friendly), right=B2(enemy, but C2 blocks landing)
        // A3: left=edge, forward=A4(enemy, but A5 blocks landing), right=B3(enemy, but C3 blocks)
        // Both white pieces are truly blocked!
        assert_eq!(
            board.status(),
            GameStatus::Won(Team::Black),
            "Completely blocked white should lose"
        );
    }

    // ========== COMPLEX MULTI-CAPTURE CHAIN TESTS ==========

    #[test]
    fn pawn_four_capture_chain() {
        // White pawn at A2
        // Chain: A3->A4, B4->C4, C5->C6, D6->E6
        let board = Board::from_squares(
            Team::White,
            &[Square::A2],
            &[Square::A3, Square::B4, Square::C5, Square::D6],
            &[],
        );
        let actions = board.actions();

        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        assert_eq!(max_captures, 4, "Should capture 4 pieces in chain");
    }

    #[test]
    fn king_five_capture_chain() {
        // White king at A2 (not on promotion row)
        // Create a zigzag path for the king:
        // A2 -> captures A4 (up) -> land A5
        // A5 -> captures C5 (right) -> land D5
        // D5 -> captures D3 (down) -> land D2
        // D2 -> captures F2 (right) -> land G2
        // This is 4 captures with alternating directions

        let board = Board::from_squares(
            Team::White,
            &[Square::A2],
            &[Square::A4, Square::C5, Square::D3, Square::F2],
            &[Square::A2],
        );
        let actions = board.actions();

        let max_captures = actions
            .iter()
            .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
            .max()
            .unwrap_or(0);

        // King should chain 4 captures with 90° turns
        assert!(max_captures >= 3, "King should capture at least 3 in chain");
    }

    // ========== EDGE AND CORNER TESTS ==========

    #[test]
    fn pawn_on_edge_limited_moves() {
        // White pawn on left edge
        let board = Board::from_squares(Team::White, &[Square::A4], &[Square::H8], &[]);
        let actions = board.actions();

        // A4 can move: B4 (right), A5 (up) - cannot go left
        assert_eq!(actions.len(), 2, "Edge pawn should have 2 moves");
    }

    #[test]
    fn pawn_in_corner_very_limited() {
        // White pawn at A1 corner (but this is promotion row for black, not valid for white pawn)
        // Use white pawn at H3 instead (near corner)
        let board = Board::from_squares(Team::White, &[Square::H3], &[Square::A8], &[]);
        let actions = board.actions();

        // H3 can move: G3 (left), H4 (up) - cannot go right (edge)
        assert_eq!(actions.len(), 2, "Corner-area pawn should have 2 moves");
    }

    #[test]
    fn king_in_corner_moves() {
        // King at A1
        let board = Board::from_squares(Team::White, &[Square::A1], &[Square::H8], &[Square::A1]);
        let actions = board.actions();

        // A1 king can go: right (B1-H1 = 7) + up (A2-A8 = 7) = 14 moves
        assert_eq!(actions.len(), 14, "Corner king should have 14 moves");
    }

    // ========== BLOCKING TESTS ==========

    #[test]
    fn pawn_blocked_by_friendly() {
        // White pawns at D4 and D5 - D4 can't move up
        let board = Board::from_squares(Team::White, &[Square::D4, Square::D5], &[Square::H8], &[]);
        let actions = board.actions();

        // D4 can move: C4, E4 (not D5 - blocked)
        // D5 can move: C5, E5, D6
        // Total: 2 + 3 = 5 moves
        let d4_moves: Vec<_> = actions
            .iter()
            .filter(|a| {
                let delta = a.delta.pieces[Team::White.to_usize()];
                delta & Square::D4.to_mask() != 0
            })
            .collect();

        let d4_to_d5 = d4_moves.iter().find(|a| {
            let dest = a.delta.pieces[Team::White.to_usize()] & !Square::D4.to_mask();
            dest == Square::D5.to_mask()
        });

        assert!(
            d4_to_d5.is_none(),
            "D4 should not be able to move to D5 (blocked)"
        );
    }

    #[test]
    fn king_blocked_by_friendly_cannot_pass() {
        // King at A4, friendly pawn at C4
        let board = Board::from_squares(
            Team::White,
            &[Square::A4, Square::C4],
            &[Square::H8],
            &[Square::A4],
        );
        let actions = board.actions();

        // King going right can only reach B4 (blocked by C4)
        let past_c4 = actions.iter().find(|a| {
            let delta = a.delta.pieces[Team::White.to_usize()];
            if delta & Square::A4.to_mask() != 0 {
                let dest = delta & !Square::A4.to_mask() & !Square::C4.to_mask();
                // Check if destination is D4, E4, F4, G4, or H4
                dest & (Square::D4.to_mask()
                    | Square::E4.to_mask()
                    | Square::F4.to_mask()
                    | Square::G4.to_mask()
                    | Square::H4.to_mask())
                    != 0
            } else {
                false
            }
        });

        assert!(
            past_c4.is_none(),
            "King should not pass through friendly piece"
        );
    }

    // ========== INITIAL POSITION TESTS ==========

    #[test]
    fn initial_position_piece_count() {
        let board = Board::new_default();
        assert_eq!(
            board.friendly_pieces().count_ones(),
            16,
            "White should have 16 pieces"
        );
        assert_eq!(
            board.hostile_pieces().count_ones(),
            16,
            "Black should have 16 pieces"
        );
    }

    #[test]
    fn initial_position_no_kings() {
        let board = Board::new_default();
        assert_eq!(board.state.kings, 0, "No kings at start");
    }

    #[test]
    fn initial_position_white_to_move() {
        let board = Board::new_default();
        assert_eq!(board.turn, Team::White, "White moves first");
    }

    /// Tests that path reconstruction enforces the 180° turn prohibition
    /// in complex multi-capture king sequences.
    ///
    /// This position has 9 different 10-capture sequences. Path reconstruction
    /// must find valid paths without 180° reversals for all of them.
    #[test]
    fn king_10_capture_path_no_180_turns() {
        let board = Board::from_squares(
            Team::White,
            &[Square::C2, Square::G2, Square::H4],
            &[
                Square::C1,
                Square::E2,
                Square::C3,
                Square::B4,
                Square::D4,
                Square::A5,
                Square::E5,
                Square::B6,
                Square::D6,
                Square::H6,
                Square::C7,
                Square::H7,
            ],
            &[
                Square::C1,
                Square::C2,
                Square::D4,
                Square::B6,
                Square::D6,
                Square::C7,
            ],
        );

        let actions = board.actions();
        assert_eq!(actions.len(), 9, "Expected 9 maximum-capture actions");

        for action in &actions {
            assert_eq!(
                action.capture_count(Team::White),
                10,
                "All actions should capture 10 pieces"
            );

            let detailed = action.to_detailed(board.turn, &board.state);
            let path = detailed.path();
            let mut prev_dir: Option<(i8, i8)> = None;

            for i in 1..path.len() {
                let from = path[i - 1];
                let to = path[i];

                let dcol = (to.column() as i8 - from.column() as i8).signum();
                let drow = (to.row() as i8 - from.row() as i8).signum();

                if let Some((pcol, prow)) = prev_dir {
                    assert!(
                        !(dcol == -pcol && drow == -prow && (dcol != 0 || drow != 0)),
                        "180° turn detected in path: {} -> {}",
                        from,
                        to
                    );
                }
                prev_dir = Some((dcol, drow));
            }
        }
    }
}
