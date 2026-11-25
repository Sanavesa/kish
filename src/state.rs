//! Bitboard state representation for Turkish Draughts.
//!
//! This module defines the [`State`] struct which stores the complete board position
//! using efficient bitboard representation.
//!
//! # Bitboard Representation
//!
//! A bitboard uses a 64-bit integer where each bit represents one square.
//! Bit 0 = A1, Bit 7 = H1, Bit 56 = A8, Bit 63 = H8.
//!
//! The state consists of three bitboards:
//! - `pieces[0]`: White pieces (men and kings)
//! - `pieces[1]`: Black pieces (men and kings)
//! - `kings`: King pieces (subset of both colors)
//!
//! # Bitboard Operations
//!
//! Common bitboard operations used in this engine:
//!
//! | Operation | Description |
//! |-----------|-------------|
//! | `mask << 8` | Shift up one row |
//! | `mask >> 8` | Shift down one row |
//! | `mask << 1` | Shift right one column (watch for wrap) |
//! | `mask >> 1` | Shift left one column (watch for wrap) |
//! | `a & b` | Intersection of positions |
//! | `a \| b` | Union of positions |
//! | `a ^ b` | Toggle positions (XOR) |
//! | `!a` | Complement (all other squares) |
//!
//! # Example
//!
//! ```rust
//! use kish::{State, Square};
//!
//! // Create initial position
//! let state = State::default();
//!
//! // White has a piece on A2
//! assert_ne!(state.pieces[0] & Square::A2.to_mask(), 0);
//!
//! // Black has a piece on A7
//! assert_ne!(state.pieces[1] & Square::A7.to_mask(), 0);
//!
//! // No kings at start
//! assert_eq!(state.kings, 0);
//! ```
//!
//! # XOR-Based State Updates
//!
//! State changes are applied using XOR operations for efficiency:
//!
//! ```rust
//! use kish::{State, Square};
//!
//! let mut state = State::zeros();
//! state.pieces[0] = Square::D4.to_mask();
//!
//! // Create a delta that moves D4 to D5
//! let delta = State::new(
//!     [Square::D4.to_mask() | Square::D5.to_mask(), 0],
//!     0,
//! );
//!
//! state.apply_(&delta);
//! assert_eq!(state.pieces[0], Square::D5.to_mask());
//! ```

use std::fmt;

use crate::Team;

// Row masks (internal use only, complete set for maintainability)
pub(crate) const MASK_ROW_1: u64 = 0x0000_0000_0000_00FF;
pub(crate) const MASK_ROW_2: u64 = 0x0000_0000_0000_FF00;
#[allow(dead_code)]
pub(crate) const MASK_ROW_3: u64 = 0x0000_0000_00FF_0000;
#[allow(dead_code)]
pub(crate) const MASK_ROW_4: u64 = 0x0000_0000_FF00_0000;
#[allow(dead_code)]
pub(crate) const MASK_ROW_5: u64 = 0x0000_00FF_0000_0000;
#[allow(dead_code)]
pub(crate) const MASK_ROW_6: u64 = 0x0000_FF00_0000_0000;
pub(crate) const MASK_ROW_7: u64 = 0x00FF_0000_0000_0000;
pub(crate) const MASK_ROW_8: u64 = 0xFF00_0000_0000_0000;

// Column masks (internal use only, complete set for maintainability)
pub(crate) const MASK_COL_A: u64 = 0x0101_0101_0101_0101;
pub(crate) const MASK_COL_B: u64 = 0x0202_0202_0202_0202;
#[allow(dead_code)]
pub(crate) const MASK_COL_C: u64 = 0x0404_0404_0404_0404;
#[allow(dead_code)]
pub(crate) const MASK_COL_D: u64 = 0x0808_0808_0808_0808;
#[allow(dead_code)]
pub(crate) const MASK_COL_E: u64 = 0x1010_1010_1010_1010;
#[allow(dead_code)]
pub(crate) const MASK_COL_F: u64 = 0x2020_2020_2020_2020;
pub(crate) const MASK_COL_G: u64 = 0x4040_4040_4040_4040;
pub(crate) const MASK_COL_H: u64 = 0x8080_8080_8080_8080;

// Promotion row masks indexed by team
pub(crate) const MASK_ROW_PROMOTIONS: [u64; 2] = [MASK_ROW_8, MASK_ROW_1];

/// The complete board state using bitboard representation.
///
/// This struct stores piece positions for both teams and king status
/// using three 64-bit integers (24 bytes total).
///
/// # Fields
///
/// - `pieces[0]`: Bitboard of all White pieces
/// - `pieces[1]`: Bitboard of all Black pieces
/// - `kings`: Bitboard of all king pieces (subset of `pieces[0] | pieces[1]`)
///
/// # Invariants
///
/// - `pieces[0] & pieces[1] == 0` (no square has both colors)
/// - `kings & (pieces[0] | pieces[1]) == kings` (kings must be on a piece)
/// - Each team has at most 16 pieces
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct State {
    /// Bitboard of pieces for each team: `pieces[0]` = White, `pieces[1]` = Black.
    pub pieces: [u64; 2],
    /// Bitboard of king pieces (men that have been promoted).
    pub kings: u64,
}

impl State {
    /// Creates a new state.
    #[must_use]
    pub const fn new(pieces: [u64; 2], kings: u64) -> Self {
        Self { pieces, kings }
    }

    /// Creates a new state with default configuration.
    #[must_use]
    pub const fn default() -> Self {
        let whites = MASK_ROW_2 | MASK_ROW_3;
        let blacks = MASK_ROW_6 | MASK_ROW_7;
        Self {
            pieces: [whites, blacks],
            kings: 0u64,
        }
    }

    /// Creates a new state with zeros.
    #[must_use]
    pub const fn zeros() -> Self {
        Self {
            pieces: [0u64, 0u64],
            kings: 0u64,
        }
    }
}

impl State {
    /// Validates the state invariants in debug builds.
    ///
    /// This method uses `debug_assert!` for performance, so checks are only
    /// performed in debug builds. In release builds, this is a no-op.
    ///
    /// # Panics (debug builds only)
    ///
    /// Panics if any invariant is violated:
    /// - A square is occupied by both teams
    /// - A king exists on an empty square
    /// - A team has more than 16 pieces
    ///
    /// # Note
    ///
    /// This does NOT validate that pieces on the promotion row are kings.
    /// During multi-capture sequences, pawns can temporarily be on the promotion
    /// row before being promoted at the end of the sequence. The promotion logic
    /// in `generate_pawn_captures_at` ensures pawns are promoted when sequences
    /// end on the promotion row.
    pub fn validate(&self) {
        debug_assert_eq!(
            self.pieces[0] & self.pieces[1],
            0,
            "a single square is occupied by both teams"
        );

        debug_assert_eq!(
            self.kings,
            self.kings & (self.pieces[0] | self.pieces[1]),
            "an empty square is marked as king"
        );

        for (team_index, pieces) in self.pieces.iter().enumerate() {
            let team = Team::from_usize(team_index);

            debug_assert!(pieces.count_ones() <= 16, "more than 16 pieces for {team}",);
        }
    }

    /// The bitboard representing the empty squares.
    #[inline(always)]
    #[must_use]
    pub const fn empty(&self) -> u64 {
        !(self.pieces[0] | self.pieces[1])
    }

    /// Applies a transformation to the state in-place.
    #[inline(always)]
    pub const fn apply_(&mut self, other: &Self) {
        self.pieces[0] ^= other.pieces[0];
        self.pieces[1] ^= other.pieces[1];
        self.kings ^= other.kings;
    }

    /// Returns a new state after applying the transformation.
    #[inline(always)]
    #[must_use]
    pub fn apply(&self, other: &Self) -> Self {
        let mut new_state = *self;
        new_state.apply_(other);
        new_state
    }

    /// Rotates the state by 180 degrees in-place.
    #[inline]
    pub const fn rotate_(&mut self) {
        let tmp_pieces = self.pieces[0];
        self.pieces[0] = self.pieces[1].reverse_bits();
        self.pieces[1] = tmp_pieces.reverse_bits();
        self.kings = self.kings.reverse_bits();
    }

    /// Returns a new state after rotating by 180 degrees.
    #[inline]
    #[must_use]
    pub fn rotate(&self) -> Self {
        let mut new_state = *self;
        new_state.rotate_();
        new_state
    }
}

impl Default for State {
    /// Returns the initial game position.
    ///
    /// This is equivalent to [`State::default()`](State::default).
    fn default() -> Self {
        Self::default()
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Add A-H column labels at the top
        let mut result = "   A B C D E F G H\n".to_string();

        for row in (0..8).rev() {
            // Add 1-8 row labels on the left
            let mut line = format!("{} ", row + 1);

            for col in 0..8 {
                let index: u8 = row * 8 + col;
                let mut ch = '.';

                if (self.pieces[0] >> index) & 1u64 == 1u64 {
                    ch = 'w';
                } else if (self.pieces[1] >> index) & 1u64 == 1u64 {
                    ch = 'b';
                }

                if (self.kings >> index) & 1u64 == 1u64 {
                    ch = ch.to_ascii_uppercase();
                }

                line += &format!(" {ch}");
            }

            // Add 1-8 row labels on the right
            line += &format!("  {}\n", row + 1);

            result += &line;
        }

        // Add A-H column labels at the bottom
        result += "   A B C D E F G H";

        write!(f, "{result}")
    }
}

#[cfg(test)]
mod tests {

    use crate::Square;

    use super::*;

    #[test]
    fn new() {
        let whites = MASK_ROW_2;
        let blacks = MASK_ROW_7;
        let kings = MASK_ROW_2;
        let state = State::new([whites, blacks], kings);
        assert_eq!(state.pieces[Team::White.to_usize()], whites);
        assert_eq!(state.pieces[Team::Black.to_usize()], blacks);
        assert_eq!(state.kings, kings);
    }

    #[test]
    fn default() {
        let state = State::default();
        assert_eq!(
            state.pieces[Team::White.to_usize()],
            MASK_ROW_2 | MASK_ROW_3
        );
        assert_eq!(
            state.pieces[Team::Black.to_usize()],
            MASK_ROW_6 | MASK_ROW_7
        );
        assert_eq!(state.kings, 0u64);
    }

    #[test]
    fn default_trait() {
        // Test that the Default trait implementation matches the inherent method
        let state_inherent = State::default();
        let state_trait: State = Default::default();
        assert_eq!(state_inherent, state_trait);
    }

    #[test]
    fn zeros() {
        let state = State::zeros();
        assert_eq!(state.pieces[Team::White.to_usize()], 0u64);
        assert_eq!(state.pieces[Team::Black.to_usize()], 0u64);
        assert_eq!(state.kings, 0u64);
    }

    #[test]
    fn validate_default() {
        let state = State::default();
        state.validate();
    }

    #[test]
    #[should_panic(expected = "a single square is occupied by both teams")]
    fn invalid_validate_common_pieces() {
        let whites = MASK_ROW_2 | MASK_ROW_3;
        let blacks = MASK_ROW_3 | MASK_ROW_4;
        let kings = 0u64;
        let state = State::new([whites, blacks], kings);
        state.validate();
    }

    #[test]
    #[should_panic(expected = "an empty square is marked as king")]
    fn invalid_validate_empty_square_as_king() {
        let whites = MASK_ROW_2;
        let blacks = MASK_ROW_3;
        let kings = MASK_ROW_1 | MASK_ROW_2;
        let state = State::new([whites, blacks], kings);
        state.validate();
    }

    #[test]
    #[should_panic(expected = "more than 16 pieces for White")]
    fn invalid_validate_more_than_16_white_pieces() {
        let whites = MASK_ROW_2 | MASK_ROW_3 | Square::A4.to_mask();
        let blacks = MASK_ROW_7;
        let kings = 0u64;
        let state = State::new([whites, blacks], kings);
        state.validate();
    }

    #[test]
    #[should_panic(expected = "more than 16 pieces for Black")]
    fn invalid_validate_more_than_16_black_pieces() {
        let whites = MASK_ROW_2;
        let blacks = MASK_ROW_6 | MASK_ROW_7 | Square::A5.to_mask();
        let kings = 0u64;
        let state = State::new([whites, blacks], kings);
        state.validate();
    }

    // Note: Removed invalid_validate_white_promotion and invalid_validate_black_promotion
    // tests. Pawns CAN temporarily be on the promotion row during multi-capture sequences.
    // Promotion is added at the end of the sequence by generate_pawn_captures_at.

    #[test]
    fn empty() {
        let state = State::default();
        assert_eq!(
            state.empty(),
            !(MASK_ROW_2 | MASK_ROW_3 | MASK_ROW_6 | MASK_ROW_7)
        );
    }

    #[test]
    fn apply() {
        let state = State::default();
        let transformation = State::new([MASK_ROW_2, MASK_ROW_7], MASK_ROW_3); // del row 2&7, promote row 3
        let new_state = state.apply(&transformation);
        assert_eq!(new_state.pieces[Team::White.to_usize()], MASK_ROW_3);
        assert_eq!(new_state.pieces[Team::Black.to_usize()], MASK_ROW_6);
        assert_eq!(new_state.kings, MASK_ROW_3);

        // Test in-place
        let mut new_state_ = state;
        new_state_.apply_(&transformation);
        assert_eq!(new_state_, new_state);
    }

    #[test]
    fn rotate() {
        let state = State::new(
            [
                MASK_ROW_2 | Square::B3.to_mask(),
                MASK_ROW_6 | Square::F5.to_mask(),
            ],
            Square::B3.to_mask() | Square::F5.to_mask(),
        );
        let expected = State::new(
            [
                MASK_ROW_3 | Square::C4.to_mask(),
                MASK_ROW_7 | Square::G6.to_mask(),
            ],
            Square::G6.to_mask() | Square::C4.to_mask(),
        );
        let new_state = state.rotate();
        assert_eq!(new_state, expected);

        // Test in-place
        let mut new_state_ = state;
        new_state_.rotate_();
        assert_eq!(new_state_, new_state);
    }

    #[test]
    fn fmt() {
        let state = State::default();
        let expected = "   A B C D E F G H\n8  . . . . . . . .  8\n7  b b b b b b b b  7\n6  b b b b b b b b  6\n5  . . . . . . . .  5\n4  . . . . . . . .  4\n3  w w w w w w w w  3\n2  w w w w w w w w  2\n1  . . . . . . . .  1\n   A B C D E F G H";
        let result = format!("{state}");
        assert_eq!(result, expected);
    }

    #[test]
    fn fmt_with_kings() {
        let state = State::new(
            [Square::A1.to_mask(), Square::H8.to_mask()],
            Square::A1.to_mask() | Square::H8.to_mask(),
        );
        let result = format!("{state}");
        assert!(result.contains('W'), "White king should display as 'W'");
        assert!(result.contains('B'), "Black king should display as 'B'");
    }
}
