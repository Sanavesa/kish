//! Move and action representations for Turkish Draughts.
//!
//! This module provides two representations for game moves:
//!
//! - [`Action`]: Compact delta-based representation for fast simulations
//! - [`ActionPath`]: Full path representation for UI and notation
//!
//! # Action (Fast Representation)
//!
//! The [`Action`] struct stores moves as XOR deltas, which allows efficient
//! application and reversal of moves during game tree search:
//!
//! ```rust
//! use kish::{Board, Team};
//!
//! let board = Board::new_default();
//! let actions = board.actions();
//!
//! // Apply action using XOR delta (very fast)
//! let new_board = board.apply(&actions[0]);
//! ```
//!
//! # ActionPath (UI Representation)
//!
//! The [`ActionPath`] struct stores the complete path of squares visited,
//! which is required for proper notation (especially multi-capture sequences):
//!
//! ```rust
//! use kish::{ActionPath, Square};
//!
//! // Create a capture action: D4 captures D5, lands on D6
//! let action = ActionPath::new_capture(Square::D4, &[Square::D6], false);
//! assert_eq!(action.to_notation(), "d4xd6");
//!
//! // Multi-capture with notation
//! let multi = ActionPath::new_capture(Square::B3, &[Square::D3, Square::D5], false);
//! assert_eq!(multi.to_notation(), "b3xd3xd5");
//! ```
//!
//! # Converting Between Representations
//!
//! Use [`Action::to_detailed`] to convert from the fast representation to the
//! detailed representation when you need notation or UI display:
//!
//! ```rust
//! use kish::{Board, Team};
//!
//! let board = Board::new_default();
//! let actions = board.actions();
//!
//! // Convert to detailed for notation
//! let detailed = actions[0].to_detailed(board.turn, &board.state);
//! println!("Move: {}", detailed.to_notation());
//! ```
//!
//! # Notation Format (Section 10)
//!
//! | Move Type | Format | Example |
//! |-----------|--------|---------|
//! | Simple move | `from-to` | `d3-d4` |
//! | Capture | `fromxto` | `d4xd6` |
//! | Multi-capture | `fromxmid...xto` | `d4xd6xf6` |
//! | Promotion | `from-to=K` | `d7-d8=K` |
//! | Capture + promotion | `fromxto=K` | `c7xc8=K` |

use std::fmt;

use super::{Square, State, Team};
use crate::state::MASK_ROW_PROMOTIONS;
#[cfg(debug_assertions)]
use crate::state::{MASK_COL_A, MASK_COL_H, MASK_ROW_1, MASK_ROW_8};

/// Maximum number of squares in a move path (source + up to 16 landing squares for captures).
/// A king can theoretically capture all 16 enemy pieces in a single chain.
const MAX_PATH_LEN: usize = 17;

/// Lookup table for file letters (a-h).
const FILE_CHARS: [u8; 8] = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h'];

/// Lookup table for rank characters (1-8).
const RANK_CHARS: [u8; 8] = [b'1', b'2', b'3', b'4', b'5', b'6', b'7', b'8'];

/// Detailed action representation with full path information for UI applications.
///
/// This stores the complete path of squares visited during a move, which is necessary
/// to generate proper algebraic notation (especially for multi-capture sequences).
/// Use this for UI display, move history, and game notation.
///
/// For high-speed simulations (perft, AI training), use [`Action`] instead.
///
/// # Notation Format (Section 10 of rules)
/// - Non-capturing move: `from-to` (e.g., `d3-d4`)
/// - Single capture: `fromxto` (e.g., `d4xd6`)
/// - Multi-capture: `fromxmidxto` (e.g., `d4xd6xf6`)
/// - Promotion: `from-to=K` (e.g., `d7-d8=K`)
/// - Capture with promotion: `fromxto=K` (e.g., `c7xc8=K`)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionPath {
    /// Path of squares: source square followed by landing squares.
    /// For a simple move: `[src, dest]`
    /// For captures: `[src, landing1, landing2, ..., final_dest]`
    path: [Square; MAX_PATH_LEN],
    /// Number of valid squares in the path.
    path_len: u8,
    /// Whether this action involves capturing.
    is_capture: bool,
    /// Whether the piece is promoted at the end.
    is_promotion: bool,
}

impl ActionPath {
    /// Creates a new non-capturing move.
    #[inline]
    #[must_use]
    pub const fn new_move(src: Square, dest: Square, is_promotion: bool) -> Self {
        let mut path = [Square::A1; MAX_PATH_LEN];
        path[0] = src;
        path[1] = dest;
        Self {
            path,
            path_len: 2,
            is_capture: false,
            is_promotion,
        }
    }

    /// Creates a new capture with a path of landing squares.
    ///
    /// # Arguments
    /// * `src` - The source square
    /// * `landings` - Slice of landing squares (where the piece lands after each capture)
    /// * `is_promotion` - Whether the piece is promoted at the end
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // MAX_PATH_LEN is 17, path_len fits in u8
    pub fn new_capture(src: Square, landings: &[Square], is_promotion: bool) -> Self {
        debug_assert!(
            !landings.is_empty(),
            "capture must have at least one landing"
        );
        debug_assert!(landings.len() < MAX_PATH_LEN, "too many landing squares");

        let mut path = [Square::A1; MAX_PATH_LEN];
        path[0] = src;
        for (i, &landing) in landings.iter().enumerate() {
            path[i + 1] = landing;
        }
        Self {
            path,
            path_len: (1 + landings.len()) as u8,
            is_capture: true,
            is_promotion,
        }
    }

    /// Returns the source square.
    #[inline]
    #[must_use]
    pub const fn source(&self) -> Square {
        self.path[0]
    }

    /// Returns the destination (final) square.
    #[inline]
    #[must_use]
    pub const fn destination(&self) -> Square {
        self.path[(self.path_len - 1) as usize]
    }

    /// Returns true if this is a capture action.
    #[inline]
    #[must_use]
    pub const fn is_capture(&self) -> bool {
        self.is_capture
    }

    /// Returns true if this action results in promotion.
    #[inline]
    #[must_use]
    pub const fn is_promotion(&self) -> bool {
        self.is_promotion
    }

    /// Returns the number of squares in the path.
    #[inline]
    #[must_use]
    pub const fn path_len(&self) -> usize {
        self.path_len as usize
    }

    /// Returns the path as a slice.
    #[inline]
    #[must_use]
    pub fn path(&self) -> &[Square] {
        &self.path[..self.path_len as usize]
    }

    /// Converts a square to its 2-character notation (e.g., "d4").
    /// Uses lookup tables for maximum performance.
    #[inline]
    const fn square_to_notation(square: Square, buf: &mut [u8; 2]) {
        let col = square.column();
        let row = square.row();
        buf[0] = FILE_CHARS[col as usize];
        buf[1] = RANK_CHARS[row as usize];
    }

    /// Converts the action to its notation string.
    ///
    /// # Performance
    /// This method is optimized to avoid heap allocations where possible by
    /// writing directly to a fixed-size buffer. The maximum notation length is:
    /// - 2 chars per square × 17 squares = 34 chars
    /// - 16 separators (- or x) = 16 chars
    /// - "=K" suffix = 2 chars
    /// - Total: 52 chars max
    #[must_use]
    pub fn to_notation(&self) -> String {
        // Pre-allocate with estimated capacity
        // Average case: 2 squares × 2 chars + 1 separator + 2 promotion = 7 chars
        let mut result = String::with_capacity(8);

        let separator = if self.is_capture { 'x' } else { '-' };
        let mut buf = [0u8; 2];

        // Write source square
        Self::square_to_notation(self.path[0], &mut buf);
        // SAFETY: buf contains valid ASCII characters from lookup tables
        result.push(buf[0] as char);
        result.push(buf[1] as char);

        // Write remaining path with separators
        for i in 1..self.path_len as usize {
            result.push(separator);
            Self::square_to_notation(self.path[i], &mut buf);
            result.push(buf[0] as char);
            result.push(buf[1] as char);
        }

        // Append promotion suffix if applicable
        if self.is_promotion {
            result.push_str("=K");
        }

        result
    }

    /// Writes the notation directly to a byte buffer without allocation.
    ///
    /// This is a performance optimization for batch processing scenarios where
    /// you need to generate many notations without heap allocations. For most
    /// use cases, prefer [`to_notation`](Self::to_notation) instead.
    ///
    /// Returns the number of bytes written.
    ///
    /// # Panics
    /// In debug builds, panics if the buffer is smaller than 52 bytes.
    ///
    /// # Example
    /// ```
    /// use kish::{ActionPath, Square};
    ///
    /// let action = ActionPath::new_move(Square::E3, Square::E4, false);
    /// let mut buf = [0u8; 52];
    /// let len = action.write_notation(&mut buf);
    /// let notation = std::str::from_utf8(&buf[..len]).unwrap();
    /// assert_eq!(notation, "e3-e4");
    /// ```
    #[inline]
    pub fn write_notation(&self, buf: &mut [u8]) -> usize {
        debug_assert!(buf.len() >= 52, "buffer too small");

        let separator = if self.is_capture { b'x' } else { b'-' };
        let mut pos = 0;

        // Write source square
        let col = self.path[0].column();
        let row = self.path[0].row();
        buf[pos] = FILE_CHARS[col as usize];
        buf[pos + 1] = RANK_CHARS[row as usize];
        pos += 2;

        // Write remaining path with separators
        for i in 1..self.path_len as usize {
            buf[pos] = separator;
            pos += 1;

            let col = self.path[i].column();
            let row = self.path[i].row();
            buf[pos] = FILE_CHARS[col as usize];
            buf[pos + 1] = RANK_CHARS[row as usize];
            pos += 2;
        }

        // Append promotion suffix if applicable
        if self.is_promotion {
            buf[pos] = b'=';
            buf[pos + 1] = b'K';
            pos += 2;
        }

        pos
    }
}

impl fmt::Display for ActionPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_notation())
    }
}

/// Fast action representation for high-speed simulations.
///
/// This is a minimal representation that stores only the bitboard state
/// changes (delta). It is optimized for high-speed simulations like perft and
/// AI model training.
///
/// For a detailed representation with full path information suitable for UI
/// applications, see [`ActionPath`]. You can convert an `Action` to an
/// `ActionPath` using [`Action::to_detailed`] with the original board state.
///
/// # Querying Action Properties
///
/// Use the provided methods to query action properties safely:
/// - [`source`](Self::source) - Get the source square
/// - [`destination`](Self::destination) - Get the destination square
/// - [`is_capture`](Self::is_capture) - Check if this is a capture
/// - [`capture_count`](Self::capture_count) - Get number of captured pieces
/// - [`captured_pieces`](Self::captured_pieces) - Get captured pieces bitboard
/// - [`is_promotion`](Self::is_promotion) - Check if this results in promotion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action {
    /// The bitboard state changes (XOR delta).
    ///
    /// This is a low-level field exposed for advanced use cases requiring
    /// direct bitboard manipulation. For most use cases, prefer the accessor
    /// methods like [`source`](Self::source), [`destination`](Self::destination),
    /// [`is_capture`](Self::is_capture), etc.
    pub delta: State,
}

impl Action {
    /// Empty action sentinel (zeroed state). Combining with this is identity.
    pub(crate) const EMPTY: Self = Self {
        delta: State::zeros(),
    };

    /// Returns true if this is an empty/sentinel action.
    #[inline]
    #[must_use]
    pub(crate) const fn is_empty(&self) -> bool {
        // Use bitwise OR to combine all fields into single comparison
        // This avoids branching from short-circuit evaluation
        (self.delta.pieces[0] | self.delta.pieces[1] | self.delta.kings) == 0
    }

    /// Creates a new action from high-level parameters.
    ///
    /// This is a convenience constructor that automatically determines whether
    /// the move is a pawn or king move, and whether it involves captures.
    ///
    /// # Arguments
    /// * `team` - The team making the move
    /// * `src_square` - The source square of the moving piece
    /// * `dest_square` - The destination square
    /// * `capture_squares` - Squares of captured pieces (empty slice for non-captures)
    /// * `kings_mask` - Bitboard of all kings on the board before the move
    ///
    /// # Example
    /// ```
    /// use kish::{Action, Square, Team};
    ///
    /// // Simple pawn move
    /// let action = Action::new(Team::White, Square::D3, Square::D4, &[], 0);
    ///
    /// // Pawn capture
    /// let capture = Action::new(Team::White, Square::D4, Square::D6, &[Square::D5], 0);
    ///
    /// // King move (source is in kings_mask)
    /// let kings = Square::D4.to_mask();
    /// let king_move = Action::new(Team::White, Square::D4, Square::D8, &[], kings);
    /// ```
    #[must_use]
    pub const fn new(
        team: Team,
        src_square: Square,
        dest_square: Square,
        capture_squares: &[Square],
        kings_mask: u64,
    ) -> Self {
        let src_mask = src_square.to_mask();
        let dest_mask = dest_square.to_mask();

        let is_src_king = kings_mask & src_mask != 0u64;

        let mut capture_mask = 0u64;
        let mut i = 0;
        while i < capture_squares.len() {
            let square = capture_squares[i];
            capture_mask |= square.to_mask();
            i += 1;
        }

        match team {
            Team::White => {
                if is_src_king {
                    if capture_mask == 0u64 {
                        Self::new_move_as_king::<0>(src_mask, dest_mask)
                    } else {
                        Self::new_capture_as_king::<0>(
                            src_mask,
                            dest_mask,
                            capture_mask,
                            kings_mask,
                        )
                    }
                } else if capture_mask == 0u64 {
                    Self::new_move_as_pawn::<0>(src_mask, dest_mask)
                } else {
                    Self::new_capture_as_pawn::<0>(src_mask, dest_mask, capture_mask, kings_mask)
                }
            }
            Team::Black => {
                if is_src_king {
                    if capture_mask == 0u64 {
                        Self::new_move_as_king::<1>(src_mask, dest_mask)
                    } else {
                        Self::new_capture_as_king::<1>(
                            src_mask,
                            dest_mask,
                            capture_mask,
                            kings_mask,
                        )
                    }
                } else if capture_mask == 0u64 {
                    Self::new_move_as_pawn::<1>(src_mask, dest_mask)
                } else {
                    Self::new_capture_as_pawn::<1>(src_mask, dest_mask, capture_mask, kings_mask)
                }
            }
        }
    }

    /// Creates a new move action for a pawn.
    #[inline]
    #[must_use]
    pub(crate) const fn new_move_as_pawn<const TEAM_INDEX: usize>(
        src_mask: u64,
        dest_mask: u64,
    ) -> Self {
        // Validate
        #[cfg(debug_assertions)]
        {
            debug_assert!(src_mask.is_power_of_two(), "source must be a single square");

            debug_assert!(
                dest_mask.is_power_of_two(),
                "destination must be a single square"
            );

            debug_assert!(
                src_mask != dest_mask,
                "source and destination cannot be the same square",
            );

            debug_assert!(
                src_mask & MASK_ROW_PROMOTIONS[TEAM_INDEX] == 0,
                "source cannot be on a promotion row",
            );

            if TEAM_INDEX == 0 {
                debug_assert!(
                    (((src_mask & !MASK_COL_A) >> 1u8) == dest_mask) // Left
                        || (((src_mask & !MASK_COL_H) << 1u8) == dest_mask) // Right
                        || (((src_mask & !MASK_ROW_8) << 8u8) == dest_mask), // Up
                    "white pawn can only move 1 unit to the left, right, or up"
                );
            } else if TEAM_INDEX == 1 {
                debug_assert!(
                    (((src_mask & !MASK_COL_A) >> 1u8) == dest_mask) // Left
                        || (((src_mask & !MASK_COL_H) << 1u8) == dest_mask) // Right
                        || (((src_mask & !MASK_ROW_1) >> 8u8) == dest_mask), // Down
                    "black pawn can only move 1 unit to the left, right, or down"
                );
            }
        }

        let mut delta = State::zeros();

        // Remove piece at source and add piece at destination
        delta.pieces[TEAM_INDEX] = src_mask ^ dest_mask;

        // Promote piece at destination if it is at the last row
        let promotion_mask = MASK_ROW_PROMOTIONS[TEAM_INDEX];
        delta.kings = dest_mask & promotion_mask;

        Self { delta }
    }

    /// Creates a new move action for a king.
    #[inline]
    #[must_use]
    pub(crate) const fn new_move_as_king<const TEAM_INDEX: usize>(
        src_mask: u64,
        dest_mask: u64,
    ) -> Self {
        // Validate
        #[cfg(debug_assertions)]
        {
            debug_assert!(src_mask.is_power_of_two(), "source must be a single square");

            debug_assert!(
                dest_mask.is_power_of_two(),
                "destination must be a single square"
            );

            debug_assert!(
                src_mask != dest_mask,
                "source and destination cannot be the same square",
            );

            // SAFETY: Both masks are verified to be single-bit by debug_asserts above.
            let src_square = unsafe { Square::from_mask(src_mask) };
            let dest_square = unsafe { Square::from_mask(dest_mask) };

            debug_assert!(
                src_square.row() == dest_square.row()
                    || src_square.column() == dest_square.column(),
                "king can only move to a square in the same row or column"
            );
        }

        let mut delta = State::zeros();

        // Remove piece at source and add piece at destination
        delta.pieces[TEAM_INDEX] = src_mask ^ dest_mask;

        // Remove king at source and add king at destination
        delta.kings = src_mask ^ dest_mask;

        Self { delta }
    }

    /// Creates a new capture action for a pawn.
    /// Note: This does NOT handle promotion. Promotion should only occur at the
    /// end of a capture sequence, not during intermediate jumps. The caller
    /// (`generate_pawn_captures_at`) is responsible for adding promotion when
    /// the sequence ends on the promotion row.
    #[inline]
    #[must_use]
    pub(crate) const fn new_capture_as_pawn<const TEAM_INDEX: usize>(
        src_mask: u64,
        dest_mask: u64,
        capture_mask: u64,
        kings_mask: u64,
    ) -> Self {
        // Validate
        #[cfg(debug_assertions)]
        {
            debug_assert!(src_mask.is_power_of_two(), "source must be a single square");

            debug_assert!(
                dest_mask.is_power_of_two(),
                "destination must be a single square"
            );

            debug_assert!(capture_mask != 0, "capture mask cannot be empty");

            // Note: pawns CAN capture from the promotion row during a multi-capture
            // sequence. They just don't become kings until the sequence ends.

            debug_assert!(
                src_mask & capture_mask == 0,
                "source cannot be a capture square"
            );
        }

        let mut delta = State::zeros();

        // Remove piece at source and add piece at destination
        delta.pieces[TEAM_INDEX] = src_mask ^ dest_mask;

        // Remove piece(s) at capture
        delta.pieces[1 - TEAM_INDEX] = capture_mask;

        // Remove king at source (if any) and remove captured kings
        // Note: Promotion is NOT handled here - it's added at the end of the
        // capture sequence by the caller.
        delta.kings = (src_mask & kings_mask) | (capture_mask & kings_mask);

        Self { delta }
    }

    /// Creates a new capture action for a king.
    #[inline]
    #[must_use]
    pub(crate) const fn new_capture_as_king<const TEAM_INDEX: usize>(
        src_mask: u64,
        dest_mask: u64,
        capture_mask: u64,
        kings_mask: u64,
    ) -> Self {
        // Validate
        #[cfg(debug_assertions)]
        {
            debug_assert!(src_mask.is_power_of_two(), "source must be a single square");

            debug_assert!(
                dest_mask.is_power_of_two(),
                "destination must be a single square"
            );

            debug_assert!(capture_mask != 0, "capture mask cannot be empty");

            debug_assert!(src_mask & kings_mask != 0, "source must be a king");

            debug_assert!(
                src_mask & capture_mask == 0,
                "source cannot be a capture square"
            );
        }

        let mut delta = State::zeros();

        // Remove piece at source and add piece at destination
        delta.pieces[TEAM_INDEX] = src_mask ^ dest_mask;

        // Remove piece(s) at capture
        delta.pieces[1 - TEAM_INDEX] = capture_mask;

        // Remove king at source and add king at destination
        // Also, remove captured kings
        delta.kings = (src_mask ^ dest_mask) | (capture_mask & kings_mask);

        Self { delta }
    }

    /// Combines two actions in-place.
    #[inline(always)]
    pub(crate) fn combine_(&mut self, other: &Self) {
        self.delta.apply_(&other.delta);
    }

    /// Returns the action after combining it with the other action.
    #[inline(always)]
    #[must_use]
    pub(crate) fn combine(&self, action: &Self) -> Self {
        let mut new_action = *self;
        new_action.combine_(action);
        new_action
    }

    /// Converts this Action to an `ActionPath`, reconstructing intermediate landing squares.
    ///
    /// This conversion extracts the source, destination, capture status, promotion status,
    /// and reconstructs the intermediate landing squares for multi-capture sequences by
    /// analyzing the captured pieces and determining the path taken.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move
    /// * `original_state` - The board state before this action was applied
    ///
    /// # Returns
    /// An `ActionPath` with full path information including intermediate landing squares.
    ///
    /// # Example
    /// ```
    /// use kish::{Action, Board, Square, Team};
    ///
    /// let board = Board::from_squares(
    ///     Team::White,
    ///     &[Square::D4],
    ///     &[Square::D5],
    ///     &[],
    /// );
    /// let actions = board.actions();
    /// let detailed = actions[0].to_detailed(board.turn, &board.state);
    /// // detailed will have path [d4, d6] for capturing d5 and landing on d6
    /// assert_eq!(detailed.to_notation(), "d4xd6");
    /// ```
    #[must_use]
    pub fn to_detailed(&self, team: Team, original_state: &State) -> ActionPath {
        let team_index = team.to_usize();
        let opponent_index = 1 - team_index;

        // The delta for our team contains src XOR dest bits
        let our_delta = self.delta.pieces[team_index];
        let original_pieces = original_state.pieces[team_index];

        // Source: bit that was set in original AND is toggled in delta
        let src_mask = our_delta & original_pieces;
        // SAFETY: src_mask is a single bit (XOR of single pieces).
        let src = unsafe { Square::from_mask(src_mask) };

        // Destination: bit that was NOT set in original AND is toggled in delta
        let dest_mask = our_delta & !original_pieces;
        // SAFETY: dest_mask is a single bit (XOR of single pieces).
        let dest = unsafe { Square::from_mask(dest_mask) };

        // Is this a capture? Check if opponent pieces are affected
        let captured_mask = self.delta.pieces[opponent_index];
        let is_capture = captured_mask != 0;

        // Is this a promotion?
        let promotion_row = MASK_ROW_PROMOTIONS[team_index];
        let was_king = (original_state.kings & src_mask) != 0;
        let is_dest_on_promotion_row = (dest_mask & promotion_row) != 0;
        let dest_in_kings_delta = (self.delta.kings & dest_mask) != 0;
        let is_promotion = !was_king && is_dest_on_promotion_row && dest_in_kings_delta;

        if !is_capture {
            return ActionPath::new_move(src, dest, is_promotion);
        }

        // For captures, reconstruct the path by finding intermediate landing squares
        let capture_count = captured_mask.count_ones() as usize;

        if capture_count == 1 {
            // Single capture: path is just [src, dest]
            return ActionPath::new_capture(src, &[dest], is_promotion);
        }

        // Multi-capture: reconstruct intermediate landing squares using backtracking.
        // Strategy: Try all possible capture sequences recursively, verifying each path leads
        // to a valid completion. This is necessary because kings have multiple valid landing
        // squares after each capture, and a greedy approach may choose an incorrect intermediate.
        let mut landings = [Square::A1; MAX_PATH_LEN - 1];
        let mut landing_count = 0;

        let found = Self::reconstruct_capture_path(
            src,
            captured_mask,
            dest_mask,
            was_king,
            team,
            None,
            &mut landings,
            &mut landing_count,
        );

        debug_assert!(
            found,
            "Failed to reconstruct valid capture path - this indicates a bug"
        );

        ActionPath::new_capture(src, &landings[..landing_count], is_promotion)
    }

    /// Recursively reconstructs the capture path using backtracking.
    /// Returns true if a valid path was found, false otherwise.
    ///
    /// # Arguments
    /// * `current` - The current position of the capturing piece
    /// * `remaining_captures` - Bitboard of remaining captured pieces to find
    /// * `final_dest_mask` - The final destination mask
    /// * `is_king` - Whether the capturing piece is a king
    /// * `team` - The team making the capture
    /// * `landings` - Output array for landing squares
    /// * `landing_count` - Output count of landing squares
    #[allow(clippy::too_many_arguments)]
    fn reconstruct_capture_path(
        current: Square,
        remaining_captures: u64,
        final_dest_mask: u64,
        is_king: bool,
        team: Team,
        prev_direction: Option<(i8, i8)>,
        landings: &mut [Square; MAX_PATH_LEN - 1],
        landing_count: &mut usize,
    ) -> bool {
        // Base case: no more captures remaining
        if remaining_captures == 0 {
            // The last landing should be the final destination
            if *landing_count > 0 {
                let last_landing = landings[*landing_count - 1];
                return last_landing.to_mask() == final_dest_mask;
            }
            return false;
        }

        let current_row = current.row() as i8;
        let current_col = current.column() as i8;

        // Directions: (row_delta, col_delta)
        const ALL_DIRECTIONS: [(i8, i8); 4] = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        const WHITE_PAWN_DIRECTIONS: [(i8, i8); 3] = [(0, 1), (0, -1), (1, 0)];
        const BLACK_PAWN_DIRECTIONS: [(i8, i8); 3] = [(0, 1), (0, -1), (-1, 0)];

        let directions: &[(i8, i8)] = if is_king {
            &ALL_DIRECTIONS
        } else if team == Team::White {
            &WHITE_PAWN_DIRECTIONS
        } else {
            &BLACK_PAWN_DIRECTIONS
        };

        for &(row_dir, col_dir) in directions {
            // Enforce 180° turn prohibition: skip if this direction reverses the previous
            if let Some((prev_row, prev_col)) = prev_direction {
                if row_dir == -prev_row && col_dir == -prev_col {
                    continue;
                }
            }

            if is_king {
                // King: find a captured piece and try all possible landing squares after it
                let mut dist = 1i8;
                let mut found_capture: Option<Square> = None;

                while dist < 8 {
                    let check_row = current_row + row_dir * dist;
                    let check_col = current_col + col_dir * dist;

                    if !(0..=7).contains(&check_row) || !(0..=7).contains(&check_col) {
                        break;
                    }

                    // SAFETY: check_row and check_col are verified to be in 0..=7.
                    let check_sq =
                        unsafe { Square::from_row_column(check_row as u8, check_col as u8) };
                    let check_mask = check_sq.to_mask();

                    if let Some(captured) = found_capture {
                        // We've passed a captured piece, this is a potential landing
                        // Skip if landing on another capture target (can't land on pieces)
                        if check_mask & remaining_captures != 0 {
                            dist += 1;
                            continue;
                        }

                        // Try this landing square
                        let old_count = *landing_count;
                        debug_assert!(
                            *landing_count < MAX_PATH_LEN - 1,
                            "landing count {} exceeds maximum path length {}",
                            *landing_count,
                            MAX_PATH_LEN - 1
                        );
                        landings[*landing_count] = check_sq;
                        *landing_count += 1;

                        let new_remaining = remaining_captures & !captured.to_mask();

                        if Self::reconstruct_capture_path(
                            check_sq,
                            new_remaining,
                            final_dest_mask,
                            is_king,
                            team,
                            Some((row_dir, col_dir)),
                            landings,
                            landing_count,
                        ) {
                            return true;
                        }

                        // Backtrack
                        *landing_count = old_count;
                    } else if check_mask & remaining_captures != 0 {
                        // Found a captured piece
                        found_capture = Some(check_sq);
                    } else {
                        // Empty square before finding a capture - can't capture in this direction
                        // from this distance, but there might be a piece further along
                    }

                    dist += 1;
                }
            } else {
                // Pawn: captures exactly 2 squares away
                let capture_row = current_row + row_dir;
                let capture_col = current_col + col_dir;
                let landing_row = current_row + row_dir * 2;
                let landing_col = current_col + col_dir * 2;

                if !(0..=7).contains(&landing_row) || !(0..=7).contains(&landing_col) {
                    continue;
                }

                // SAFETY: bounds checked above
                let capture_sq =
                    unsafe { Square::from_row_column(capture_row as u8, capture_col as u8) };
                let landing_sq =
                    unsafe { Square::from_row_column(landing_row as u8, landing_col as u8) };

                if capture_sq.to_mask() & remaining_captures != 0 {
                    let old_count = *landing_count;
                    debug_assert!(
                        *landing_count < MAX_PATH_LEN - 1,
                        "landing count {} exceeds maximum path length {}",
                        *landing_count,
                        MAX_PATH_LEN - 1
                    );
                    landings[*landing_count] = landing_sq;
                    *landing_count += 1;

                    let new_remaining = remaining_captures & !capture_sq.to_mask();

                    if Self::reconstruct_capture_path(
                        landing_sq,
                        new_remaining,
                        final_dest_mask,
                        is_king,
                        team,
                        Some((row_dir, col_dir)),
                        landings,
                        landing_count,
                    ) {
                        return true;
                    }

                    // Backtrack
                    *landing_count = old_count;
                }
            }
        }

        false
    }

    /// Returns the source square of this action.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move
    /// * `original_pieces` - The bitboard of the team's pieces before this action
    #[inline]
    #[must_use]
    pub const fn source(&self, team: Team, original_pieces: u64) -> Square {
        let team_index = team.to_usize();
        let our_delta = self.delta.pieces[team_index];
        // Source: bit that was set in original AND is toggled in delta
        let src_mask = our_delta & original_pieces;
        // SAFETY: src_mask is a single bit (action moves exactly one piece).
        unsafe { Square::from_mask(src_mask) }
    }

    /// Returns the destination square of this action.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move
    /// * `original_pieces` - The bitboard of the team's pieces before this action
    #[inline]
    #[must_use]
    pub const fn destination(&self, team: Team, original_pieces: u64) -> Square {
        let team_index = team.to_usize();
        let our_delta = self.delta.pieces[team_index];
        // Destination: bit that was NOT set in original AND is toggled in delta
        let dest_mask = our_delta & !original_pieces;
        // SAFETY: dest_mask is a single bit (action moves to exactly one square).
        unsafe { Square::from_mask(dest_mask) }
    }

    /// Returns true if this action is a capture.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move
    #[inline]
    #[must_use]
    pub const fn is_capture(&self, team: Team) -> bool {
        let opponent_index = 1 - team.to_usize();
        self.delta.pieces[opponent_index] != 0
    }

    /// Returns the number of pieces captured by this action.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move
    #[inline]
    #[must_use]
    pub const fn capture_count(&self, team: Team) -> u32 {
        let opponent_index = 1 - team.to_usize();
        self.delta.pieces[opponent_index].count_ones()
    }

    /// Returns the captured pieces as a bitboard.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move (the opponent's pieces are captured)
    #[inline]
    #[must_use]
    pub const fn captured_pieces(&self, team: Team) -> u64 {
        let opponent_index = 1 - team.to_usize();
        self.delta.pieces[opponent_index]
    }

    /// Returns true if this action results in a promotion.
    ///
    /// # Arguments
    /// * `team` - The team that is making the move
    /// * `original_state` - The board state before this action was applied
    #[inline]
    #[must_use]
    pub const fn is_promotion(&self, team: Team, original_state: &State) -> bool {
        let team_index = team.to_usize();
        let our_delta = self.delta.pieces[team_index];
        let original_pieces = original_state.pieces[team_index];

        // Source and destination masks
        let src_mask = our_delta & original_pieces;
        let dest_mask = our_delta & !original_pieces;

        // Check promotion conditions
        let promotion_row = MASK_ROW_PROMOTIONS[team_index];
        let was_king = (original_state.kings & src_mask) != 0;
        let is_dest_on_promotion_row = (dest_mask & promotion_row) != 0;
        let dest_in_kings_delta = (self.delta.kings & dest_mask) != 0;

        !was_king && is_dest_on_promotion_row && dest_in_kings_delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_case::test_case;

    #[test]
    fn move_as_white_pawn() {
        // Move white pawn from A4 to A5
        let action = Action::new_move_as_pawn::<0>(Square::A4.to_mask(), Square::A5.to_mask());

        // Whites should move a pawn from A4 to A5
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::A4.to_mask() | Square::A5.to_mask()
        );

        // Blacks and kings shouldn't be modified
        debug_assert_eq!(action.delta.pieces[Team::Black.to_usize()], 0u64);
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn move_as_white_pawn_gets_promoted() {
        // Move white pawn from C7 to C8
        let action = Action::new_move_as_pawn::<0>(Square::C7.to_mask(), Square::C8.to_mask());

        // Whites should move a pawn from C7 to C8
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::C7.to_mask() | Square::C8.to_mask()
        );

        // Blacks shouldn't be modified
        debug_assert_eq!(action.delta.pieces[Team::Black.to_usize()], 0u64);

        // Kings should promote C8
        debug_assert_eq!(action.delta.kings, Square::C8.to_mask());
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_move_as_white_pawn_src_must_be_single_square() {
        let _ = Action::new_move_as_pawn::<0>(
            Square::A2.to_mask() | Square::A3.to_mask(),
            Square::A4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_move_as_white_pawn_dest_must_be_single_square() {
        let _ = Action::new_move_as_pawn::<0>(
            Square::A3.to_mask(),
            Square::A4.to_mask() | Square::A5.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "source and destination cannot be the same square")]
    fn invalid_move_as_white_pawn_src_same_as_dest() {
        // Move white pawn from A4 to A4
        let _ = Action::new_move_as_pawn::<0>(Square::A4.to_mask(), Square::A4.to_mask());
    }

    #[test]
    #[should_panic(expected = "source cannot be on a promotion row")]
    fn invalid_move_as_white_pawn_src_at_promotion_row() {
        // Move white pawn from C8 to B8
        let _ = Action::new_move_as_pawn::<0>(Square::C8.to_mask(), Square::B8.to_mask());
    }

    #[test_case(Square::C4, Square::C3; "move down")]
    #[test_case(Square::C4, Square::C2; "move far down")]
    #[test_case(Square::B4, Square::D4; "move far left")]
    #[test_case(Square::A6, Square::C6; "move far right")]
    #[test_case(Square::D5, Square::C6; "move diagonal left")]
    #[test_case(Square::D5, Square::E6; "move diagonal right")]
    #[should_panic(expected = "white pawn can only move 1 unit to the left, right, or up")]
    fn invalid_move_as_white_pawn(src_square: Square, dest_square: Square) {
        let _ = Action::new_move_as_pawn::<0>(src_square.to_mask(), dest_square.to_mask());
    }

    #[test]
    fn move_as_black_pawn() {
        // Move black pawn from B7 to B6
        let action = Action::new_move_as_pawn::<1>(Square::B7.to_mask(), Square::B6.to_mask());

        // Blacks should move a pawn from B7 to B6
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B7.to_mask() | Square::B6.to_mask()
        );

        // Whites and kings shouldn't be modified
        debug_assert_eq!(action.delta.pieces[Team::White.to_usize()], 0u64);
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn move_as_black_pawn_gets_promoted() {
        // Move black pawn from D2 to D1
        let action = Action::new_move_as_pawn::<1>(Square::D2.to_mask(), Square::D1.to_mask());

        // Blacks should move a pawn from D2 to D1
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::D2.to_mask() | Square::D1.to_mask()
        );

        // Whites shouldn't be modified
        debug_assert_eq!(action.delta.pieces[Team::White.to_usize()], 0u64);

        // Kings should promote D1
        debug_assert_eq!(action.delta.kings, Square::D1.to_mask());
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_move_as_black_pawn_src_must_be_single_square() {
        let _ = Action::new_move_as_pawn::<1>(
            Square::D6.to_mask() | Square::D7.to_mask(),
            Square::D5.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_move_as_black_pawn_dest_must_be_single_square() {
        let _ = Action::new_move_as_pawn::<1>(
            Square::D6.to_mask(),
            Square::D4.to_mask() | Square::D5.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "source and destination cannot be the same square")]
    fn invalid_move_as_black_pawn_src_same_as_dest() {
        // Move black pawn from C5 to C5
        let _ = Action::new_move_as_pawn::<1>(Square::C5.to_mask(), Square::C5.to_mask());
    }

    #[test]
    #[should_panic(expected = "source cannot be on a promotion row")]
    fn invalid_move_as_black_pawn_src_at_promotion_row() {
        // Move black pawn from C1 to B1
        let _ = Action::new_move_as_pawn::<1>(Square::C1.to_mask(), Square::B1.to_mask());
    }

    #[test_case(Square::C4, Square::C5; "move up")]
    #[test_case(Square::C4, Square::C2; "move far down")]
    #[test_case(Square::B4, Square::D4; "move far left")]
    #[test_case(Square::A6, Square::C6; "move far right")]
    #[test_case(Square::D5, Square::C4; "move diagonal left")]
    #[test_case(Square::D5, Square::E4; "move diagonal right")]
    #[should_panic(expected = "black pawn can only move 1 unit to the left, right, or down")]
    fn invalid_move_as_black_pawn(src_square: Square, dest_square: Square) {
        let _ = Action::new_move_as_pawn::<1>(src_square.to_mask(), dest_square.to_mask());
    }

    #[test]
    fn move_as_white_king() {
        // Move white king from B5 to B1
        let action = Action::new_move_as_king::<0>(Square::B5.to_mask(), Square::B1.to_mask());

        // Whites should move from B5 to B1
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B5.to_mask() | Square::B1.to_mask()
        );

        // King should be moved from B5 to B1
        debug_assert_eq!(
            action.delta.kings,
            Square::B5.to_mask() | Square::B1.to_mask()
        );

        // Blacks shouldn't be modified
        debug_assert_eq!(action.delta.pieces[Team::Black.to_usize()], 0u64);
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_move_as_white_king_src_must_be_single_square() {
        let _ = Action::new_move_as_king::<0>(
            Square::A2.to_mask() | Square::A3.to_mask(),
            Square::A4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_move_as_white_king_dest_must_be_single_square() {
        let _ = Action::new_move_as_king::<0>(
            Square::A3.to_mask(),
            Square::A4.to_mask() | Square::A5.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "source and destination cannot be the same square")]
    fn invalid_move_as_white_king_src_same_as_dest() {
        // Move white king from B7 to B7
        let _ = Action::new_move_as_king::<0>(Square::B7.to_mask(), Square::B7.to_mask());
    }

    #[test_case(Square::A2, Square::B3; "diagonal")]
    #[test_case(Square::F1, Square::G6; "far diagonal")]
    #[should_panic(expected = "king can only move to a square in the same row or column")]
    fn invalid_move_as_white_king(src_square: Square, dest_square: Square) {
        let _ = Action::new_move_as_king::<0>(src_square.to_mask(), dest_square.to_mask());
    }

    #[test]
    fn new_move_as_king_black() {
        // Move black king from H5 to A5
        let action = Action::new_move_as_king::<1>(Square::H5.to_mask(), Square::A5.to_mask());

        // Blacks should move from H5 to A5
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::H5.to_mask() | Square::A5.to_mask()
        );

        // King should be moved from H5 to A5
        debug_assert_eq!(
            action.delta.kings,
            Square::H5.to_mask() | Square::A5.to_mask()
        );

        // Whites shouldn't be modified
        debug_assert_eq!(action.delta.pieces[Team::White.to_usize()], 0u64);
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_move_as_black_king_src_must_be_single_square() {
        let _ = Action::new_move_as_king::<1>(
            Square::D6.to_mask() | Square::D7.to_mask(),
            Square::D5.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_move_as_black_king_dest_must_be_single_square() {
        let _ = Action::new_move_as_king::<1>(
            Square::D6.to_mask(),
            Square::D4.to_mask() | Square::D5.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "source and destination cannot be the same square")]
    fn invalid_move_as_black_king_src_same_as_dest() {
        // Move black king from F3 to F3
        let _ = Action::new_move_as_king::<1>(Square::F3.to_mask(), Square::F3.to_mask());
    }

    #[test_case(Square::A4, Square::B3; "diagonal")]
    #[test_case(Square::H8, Square::C4; "far diagonal")]
    #[should_panic(expected = "king can only move to a square in the same row or column")]
    fn invalid_move_as_black_king(src_square: Square, dest_square: Square) {
        let _ = Action::new_move_as_king::<1>(src_square.to_mask(), dest_square.to_mask());
    }

    #[test]
    fn capture_as_white_pawn() {
        // White pawn at A4 captures black pawn at A5, landing at A6
        let action = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            0u64,
        );

        // Whites should move from A4 to A6
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::A4.to_mask() | Square::A6.to_mask()
        );

        // Blacks should delete a pawn at A5
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::A5.to_mask()
        );

        // Kings shouldnt be modified
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn capture_as_white_pawn_multiple_captures() {
        // White pawn at A4 captures black pawn at A5, landing at A6
        // Then captures black pawn at B6, landing at C6
        let action = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::C6.to_mask(),
            Square::A5.to_mask() | Square::B6.to_mask(),
            0u64,
        );

        // Whites should move from A4 to C6
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::A4.to_mask() | Square::C6.to_mask()
        );

        // Blacks should delete a pawn at A5 and B6
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::A5.to_mask() | Square::B6.to_mask()
        );

        // Kings shouldnt be modified
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn new_capture_as_pawn_white_lands_on_promotion_row() {
        // White pawn at B6 captures black pawn at B7, landing at B8
        // Note: Promotion is NOT handled by new_capture_as_pawn - it's added
        // at the end of the capture sequence by generate_pawn_captures_at
        let action = Action::new_capture_as_pawn::<0>(
            Square::B6.to_mask(),
            Square::B8.to_mask(),
            Square::B7.to_mask(),
            0u64,
        );

        // Whites should move from B6 to B8
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B6.to_mask() | Square::B8.to_mask()
        );

        // Blacks should delete a pawn at B7
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B7.to_mask()
        );

        // Kings should NOT be modified (promotion is handled at sequence end)
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn new_capture_as_pawn_white_removes_captured_kings() {
        // White pawn at B6 captures black king at B7, landing at B8
        // Note: Promotion is NOT handled here - only captured king removal
        let action = Action::new_capture_as_pawn::<0>(
            Square::B6.to_mask(),
            Square::B8.to_mask(),
            Square::B7.to_mask(),
            Square::B7.to_mask(),
        );

        // Whites should move from B6 to B8
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B6.to_mask() | Square::B8.to_mask()
        );

        // Blacks should delete a piece at B7
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B7.to_mask()
        );

        // Kings should only delete the captured king at B7 (no promotion)
        debug_assert_eq!(action.delta.kings, Square::B7.to_mask());
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_capture_as_white_pawn_src_must_be_single_square() {
        let _ = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask() | Square::B4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            0u64,
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_capture_as_white_pawn_dest_must_be_single_square() {
        let _ = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask() | Square::B6.to_mask(),
            Square::A5.to_mask(),
            0u64,
        );
    }

    #[test]
    fn capture_as_white_pawn_from_promotion_row() {
        // White pawn at C8 captures black pawn at B8, landing at A8
        // This is valid during multi-capture sequences where the pawn
        // temporarily lands on the promotion row before continuing
        let action = Action::new_capture_as_pawn::<0>(
            Square::C8.to_mask(),
            Square::A8.to_mask(),
            Square::B8.to_mask(),
            0u64,
        );

        // Whites should move from C8 to A8
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::C8.to_mask() | Square::A8.to_mask()
        );

        // Blacks should delete a pawn at B8
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B8.to_mask()
        );

        // Kings should NOT be modified (promotion is handled at sequence end)
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    #[should_panic(expected = "source cannot be a capture square")]
    fn invalid_capture_as_white_pawn_src_cannot_be_capture_square() {
        let _ = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A4.to_mask(),
            0u64,
        );
    }

    #[test]
    #[should_panic(expected = "capture mask cannot be empty")]
    fn invalid_capture_as_white_pawn_no_captures() {
        let _ = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            0u64,
            0u64,
        );
    }

    #[test]
    fn capture_as_black_pawn() {
        // Black pawn at B6 captures white pawn at B5, landing at B4
        let action = Action::new_capture_as_pawn::<1>(
            Square::B6.to_mask(),
            Square::B4.to_mask(),
            Square::B5.to_mask(),
            0u64,
        );

        // Blacks should move from B6 to B4
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B6.to_mask() | Square::B4.to_mask()
        );

        // Whites should delete a pawn at B5
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B5.to_mask()
        );

        // Kings shouldnt be modified
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn capture_as_black_pawn_multiple_captures() {
        // Black pawn at B6 captures white pawn at B5, landing at B4
        // Then captures white pawn at C4, landing at D4
        let action = Action::new_capture_as_pawn::<1>(
            Square::B6.to_mask(),
            Square::D4.to_mask(),
            Square::B5.to_mask() | Square::C4.to_mask(),
            0u64,
        );

        // Blacks should move from B6 to D4
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B6.to_mask() | Square::D4.to_mask()
        );

        // Whites should delete a pawn at B5 and C4
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B5.to_mask() | Square::C4.to_mask()
        );

        // Kings shouldnt be modified
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn new_capture_as_pawn_black_lands_on_promotion_row() {
        // Black pawn at B3 captures white pawn at B2, landing at B1
        // Note: Promotion is NOT handled by new_capture_as_pawn - it's added
        // at the end of the capture sequence by generate_pawn_captures_at
        let action = Action::new_capture_as_pawn::<1>(
            Square::B3.to_mask(),
            Square::B1.to_mask(),
            Square::B2.to_mask(),
            0u64,
        );

        // Black should move from B3 to B1
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B3.to_mask() | Square::B1.to_mask()
        );

        // Whites should delete a pawn at B2
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B2.to_mask()
        );

        // Kings should NOT be modified (promotion is handled at sequence end)
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    fn new_capture_as_pawn_black_removes_captured_kings() {
        // Black pawn at B3 captures white king at B2, landing at B1
        // Note: Promotion is NOT handled here - only captured king removal
        let action = Action::new_capture_as_pawn::<1>(
            Square::B3.to_mask(),
            Square::B1.to_mask(),
            Square::B2.to_mask(),
            Square::B2.to_mask(),
        );

        // Black should move from B3 to B1
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::B3.to_mask() | Square::B1.to_mask()
        );

        // Whites should delete a piece at B2
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::B2.to_mask()
        );

        // Kings should only delete the captured king at B2 (no promotion)
        debug_assert_eq!(action.delta.kings, Square::B2.to_mask());
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_capture_as_black_pawn_src_must_be_single_square() {
        let _ = Action::new_capture_as_pawn::<1>(
            Square::B6.to_mask() | Square::C6.to_mask(),
            Square::B4.to_mask(),
            Square::B5.to_mask(),
            0u64,
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_capture_as_black_pawn_dest_must_be_single_square() {
        let _ = Action::new_capture_as_pawn::<1>(
            Square::B6.to_mask(),
            Square::B4.to_mask() | Square::D4.to_mask(),
            Square::B5.to_mask(),
            0u64,
        );
    }

    #[test]
    fn capture_as_black_pawn_from_promotion_row() {
        // Black pawn at E1 captures white pawn at D1, landing at C1
        // This is valid during multi-capture sequences where the pawn
        // temporarily lands on the promotion row before continuing
        let action = Action::new_capture_as_pawn::<1>(
            Square::E1.to_mask(),
            Square::C1.to_mask(),
            Square::D1.to_mask(),
            0u64,
        );

        // Black should move from E1 to C1
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::E1.to_mask() | Square::C1.to_mask()
        );

        // Whites should delete a pawn at D1
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::D1.to_mask()
        );

        // Kings should NOT be modified (promotion is handled at sequence end)
        debug_assert_eq!(action.delta.kings, 0u64);
    }

    #[test]
    #[should_panic(expected = "source cannot be a capture square")]
    fn invalid_capture_as_black_pawn_src_cannot_be_capture_square() {
        let _ = Action::new_capture_as_pawn::<1>(
            Square::F6.to_mask(),
            Square::F4.to_mask(),
            Square::F6.to_mask(),
            0u64,
        );
    }

    #[test]
    #[should_panic(expected = "capture mask cannot be empty")]
    fn invalid_capture_as_black_pawn_no_captures() {
        let _ = Action::new_capture_as_pawn::<1>(
            Square::G6.to_mask(),
            Square::G4.to_mask(),
            0u64,
            0u64,
        );
    }

    #[test]
    fn capture_as_white_king() {
        // White king at A4 captures black pawn at D4, landing at H4
        let action = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::H4.to_mask(),
            Square::D4.to_mask(),
            Square::A4.to_mask(),
        );

        // Whites should move from A4 to H4
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::A4.to_mask() | Square::H4.to_mask()
        );

        // Blacks should delete a pawn at D4
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::D4.to_mask()
        );

        // Kings should move from A4 to H4
        debug_assert_eq!(
            action.delta.kings,
            Square::A4.to_mask() | Square::H4.to_mask()
        );
    }

    #[test]
    fn capture_as_white_king_multiple_captures() {
        // White king at A4 captures black pawn at D4, landing at H4
        // Then captures black king at H2, landing at H1
        let action = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::H1.to_mask(),
            Square::D4.to_mask() | Square::H2.to_mask(),
            Square::A4.to_mask() | Square::H2.to_mask(),
        );

        // Whites should move from A4 to H1
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::A4.to_mask() | Square::H1.to_mask()
        );

        // Blacks should delete a pawn at D4 and H2
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::D4.to_mask() | Square::H2.to_mask()
        );

        // Kings should move from A4 to H1, and remove H2
        debug_assert_eq!(
            action.delta.kings,
            Square::A4.to_mask() | Square::H1.to_mask() | Square::H2.to_mask()
        );
    }

    #[test]
    fn capture_as_white_king_removes_captured_kings() {
        // White king at F5 captures black king at F7, landing at F8
        let action = Action::new_capture_as_king::<0>(
            Square::F5.to_mask(),
            Square::F8.to_mask(),
            Square::F7.to_mask(),
            Square::F5.to_mask() | Square::F7.to_mask(),
        );

        // Whites should move from F5 to F8
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::F5.to_mask() | Square::F8.to_mask()
        );

        // Blacks should delete a pawn at F7
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::F7.to_mask()
        );

        // Kings should move from F5 to F8, and remove F7
        debug_assert_eq!(
            action.delta.kings,
            Square::F5.to_mask() | Square::F8.to_mask() | Square::F7.to_mask()
        );
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_capture_as_white_king_src_must_be_single_square() {
        let _ = Action::new_capture_as_king::<0>(
            Square::A4.to_mask() | Square::B4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            Square::A4.to_mask() | Square::B4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_capture_as_white_king_dest_must_be_single_square() {
        let _ = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask() | Square::B6.to_mask(),
            Square::A5.to_mask(),
            Square::A4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "source cannot be a capture square")]
    fn invalid_capture_as_white_king_src_cannot_be_capture_square() {
        let _ = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A4.to_mask(),
            Square::A4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "capture mask cannot be empty")]
    fn invalid_capture_as_white_king_no_captures() {
        let _ = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            0u64,
            Square::A4.to_mask(),
        );
    }

    #[test]
    fn capture_as_black_king() {
        // Black king at A4 captures white pawn at D4, landing at H4
        let action = Action::new_capture_as_king::<1>(
            Square::A4.to_mask(),
            Square::H4.to_mask(),
            Square::D4.to_mask(),
            Square::A4.to_mask(),
        );

        // Blacks should move from A4 to H4
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::A4.to_mask() | Square::H4.to_mask()
        );

        // Whites should delete a pawn at D4
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::D4.to_mask()
        );

        // Kings should move from A4 to H4
        debug_assert_eq!(
            action.delta.kings,
            Square::A4.to_mask() | Square::H4.to_mask()
        );
    }

    #[test]
    fn capture_as_black_king_multiple_captures() {
        // Black king at A4 captures white pawn at D4, landing at H4
        // Then captures white king at H2, landing at H1
        let action = Action::new_capture_as_king::<1>(
            Square::A4.to_mask(),
            Square::H1.to_mask(),
            Square::D4.to_mask() | Square::H2.to_mask(),
            Square::A4.to_mask() | Square::H2.to_mask(),
        );

        // Blacks should move from A4 to H1
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::A4.to_mask() | Square::H1.to_mask()
        );

        // Whites should delete a pawn at D4 and H2
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::D4.to_mask() | Square::H2.to_mask()
        );

        // Kings should move from A4 to H1, and remove H2
        debug_assert_eq!(
            action.delta.kings,
            Square::A4.to_mask() | Square::H1.to_mask() | Square::H2.to_mask()
        );
    }

    #[test]
    fn capture_as_black_king_removes_captured_kings() {
        // Black king at F5 captures white king at F7, landing at F8
        let action = Action::new_capture_as_king::<1>(
            Square::F5.to_mask(),
            Square::F8.to_mask(),
            Square::F7.to_mask(),
            Square::F5.to_mask() | Square::F7.to_mask(),
        );

        // Blacks should move from F5 to F8
        debug_assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::F5.to_mask() | Square::F8.to_mask()
        );

        // Whites should delete a pawn at F7
        debug_assert_eq!(
            action.delta.pieces[Team::White.to_usize()],
            Square::F7.to_mask()
        );

        // Kings should move from F5 to F8, and remove F7
        debug_assert_eq!(
            action.delta.kings,
            Square::F5.to_mask() | Square::F8.to_mask() | Square::F7.to_mask()
        );
    }

    #[test]
    #[should_panic(expected = "source must be a single square")]
    fn invalid_capture_as_black_king_src_must_be_single_square() {
        let _ = Action::new_capture_as_king::<1>(
            Square::A4.to_mask() | Square::B4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            Square::A4.to_mask() | Square::B4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "destination must be a single square")]
    fn invalid_capture_as_black_king_dest_must_be_single_square() {
        let _ = Action::new_capture_as_king::<1>(
            Square::A4.to_mask(),
            Square::A6.to_mask() | Square::B6.to_mask(),
            Square::A5.to_mask(),
            Square::A4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "source cannot be a capture square")]
    fn invalid_capture_as_black_king_src_cannot_be_capture_square() {
        let _ = Action::new_capture_as_king::<1>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A4.to_mask(),
            Square::A4.to_mask(),
        );
    }

    #[test]
    #[should_panic(expected = "capture mask cannot be empty")]
    fn invalid_capture_as_black_king_no_captures() {
        let _ = Action::new_capture_as_king::<1>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            0u64,
            Square::A4.to_mask(),
        );
    }

    #[test]
    fn combine() {
        // Move white pawn from A4 to A6, capturing black king at A5
        let action1 = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            Square::A5.to_mask(),
        );

        // Move white pawn from A6 to C6, capturing black pawn at B6
        let action2 = Action::new_capture_as_pawn::<0>(
            Square::A6.to_mask(),
            Square::C6.to_mask(),
            Square::B6.to_mask(),
            0u64,
        );

        let combined = action1.combine(&action2);

        // Whites should move a piece from A4 to C6
        debug_assert_eq!(
            combined.delta.pieces[Team::White.to_usize()],
            Square::A4.to_mask() | Square::C6.to_mask()
        );

        // Blacks should remove piece at A5 and B6
        debug_assert_eq!(
            combined.delta.pieces[Team::Black.to_usize()],
            Square::A5.to_mask() | Square::B6.to_mask()
        );

        // Kings should remove king at A5
        debug_assert_eq!(combined.delta.kings, Square::A5.to_mask());

        // Test in-place
        let mut combined_ = action1;
        combined_.combine_(&action2);
        debug_assert_eq!(combined_, combined);
    }

    // ActionNotation tests

    #[test]
    fn notation_simple_move() {
        // e3-e4: White man moves forward from e3 to e4
        let notation = ActionPath::new_move(Square::E3, Square::E4, false);
        assert_eq!(notation.to_notation(), "e3-e4");
        assert_eq!(notation.source(), Square::E3);
        assert_eq!(notation.destination(), Square::E4);
        assert!(!notation.is_capture());
        assert!(!notation.is_promotion());
        assert_eq!(notation.path_len(), 2);
    }

    #[test]
    fn notation_sideways_move() {
        // d4-e4: Man moves right from d4 to e4
        let notation = ActionPath::new_move(Square::D4, Square::E4, false);
        assert_eq!(notation.to_notation(), "d4-e4");
    }

    #[test]
    fn notation_king_long_move() {
        // a1-a7: King moves from a1 to a7 (six squares forward)
        let notation = ActionPath::new_move(Square::A1, Square::A7, false);
        assert_eq!(notation.to_notation(), "a1-a7");
    }

    #[test]
    fn notation_promotion() {
        // c7-c8=K: White man reaches c8 and is promoted to king
        let notation = ActionPath::new_move(Square::C7, Square::C8, true);
        assert_eq!(notation.to_notation(), "c7-c8=K");
        assert!(notation.is_promotion());
        assert!(!notation.is_capture());
    }

    #[test]
    fn notation_single_capture() {
        // d4xd6: Piece on d4 jumps over enemy on d5, landing on d6
        let notation = ActionPath::new_capture(Square::D4, &[Square::D6], false);
        assert_eq!(notation.to_notation(), "d4xd6");
        assert!(notation.is_capture());
        assert!(!notation.is_promotion());
        assert_eq!(notation.path_len(), 2);
    }

    #[test]
    fn notation_multi_capture() {
        // b3xd3xd5: Piece captures two enemies: first moving right (b3 to d3), then forward (d3 to d5)
        let notation = ActionPath::new_capture(Square::B3, &[Square::D3, Square::D5], false);
        assert_eq!(notation.to_notation(), "b3xd3xd5");
        assert!(notation.is_capture());
        assert_eq!(notation.path_len(), 3);
        assert_eq!(notation.path(), &[Square::B3, Square::D3, Square::D5]);
    }

    #[test]
    fn notation_capture_with_promotion() {
        // c6xc8=K: Piece captures and promotes
        let notation = ActionPath::new_capture(Square::C6, &[Square::C8], true);
        assert_eq!(notation.to_notation(), "c6xc8=K");
        assert!(notation.is_capture());
        assert!(notation.is_promotion());
    }

    #[test]
    fn notation_complex_multi_capture() {
        // a1xa3xa5xa7: Multi-capture with 3 captures
        let notation =
            ActionPath::new_capture(Square::A1, &[Square::A3, Square::A5, Square::A7], false);
        assert_eq!(notation.to_notation(), "a1xa3xa5xa7");
        assert_eq!(notation.path_len(), 4);
    }

    #[test]
    fn notation_complex_multi_capture_with_promotion() {
        // b2xb4xd4xd6xd8=K: Complex multi-capture ending in promotion
        let notation = ActionPath::new_capture(
            Square::B2,
            &[Square::B4, Square::D4, Square::D6, Square::D8],
            true,
        );
        assert_eq!(notation.to_notation(), "b2xb4xd4xd6xd8=K");
        assert!(notation.is_promotion());
    }

    #[test]
    fn notation_display_trait() {
        let notation = ActionPath::new_move(Square::D3, Square::D4, false);
        assert_eq!(format!("{notation}"), "d3-d4");
    }

    #[test]
    fn notation_write_to_buffer() {
        let notation = ActionPath::new_capture(Square::B3, &[Square::D3, Square::D5], false);
        let mut buf = [0u8; 52];
        let len = notation.write_notation(&mut buf);
        assert_eq!(len, 8); // "b3xd3xd5" = 8 chars (2+1+2+1+2)
        assert_eq!(&buf[..len], b"b3xd3xd5");
    }

    #[test]
    fn notation_write_to_buffer_with_promotion() {
        let notation = ActionPath::new_move(Square::C7, Square::C8, true);
        let mut buf = [0u8; 52];
        let len = notation.write_notation(&mut buf);
        assert_eq!(len, 7); // "c7-c8=K" = 7 chars (2+1+2+2)
        assert_eq!(&buf[..len], b"c7-c8=K");
    }

    #[test]
    fn notation_all_corners() {
        // Test all corner squares to ensure file/rank mapping is correct
        let a1 = ActionPath::new_move(Square::A1, Square::A2, false);
        assert_eq!(a1.to_notation(), "a1-a2");

        let h1 = ActionPath::new_move(Square::H1, Square::H2, false);
        assert_eq!(h1.to_notation(), "h1-h2");

        let a8 = ActionPath::new_move(Square::A7, Square::A8, false);
        assert_eq!(a8.to_notation(), "a7-a8");

        let h8 = ActionPath::new_move(Square::H7, Square::H8, false);
        assert_eq!(h8.to_notation(), "h7-h8");
    }

    #[test_case(Square::A1, "a1"; "a1")]
    #[test_case(Square::B2, "b2"; "b2")]
    #[test_case(Square::C3, "c3"; "c3")]
    #[test_case(Square::D4, "d4"; "d4")]
    #[test_case(Square::E5, "e5"; "e5")]
    #[test_case(Square::F6, "f6"; "f6")]
    #[test_case(Square::G7, "g7"; "g7")]
    #[test_case(Square::H8, "h8"; "h8")]
    fn notation_square_format(square: Square, expected_prefix: &str) {
        let notation = ActionPath::new_move(square, Square::A1, false);
        assert!(notation.to_notation().starts_with(expected_prefix));
    }

    // =====================================================================
    // Action to ActionPath conversion tests
    // =====================================================================

    #[test]
    fn action_to_notation_simple_move() {
        // White pawn moves from D4 to D5
        let original_state = State::new([Square::D4.to_mask(), Square::H8.to_mask()], 0);
        let action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::D5.to_mask());

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::D4);
        assert_eq!(notation.destination(), Square::D5);
        assert!(!notation.is_capture());
        assert!(!notation.is_promotion());
        assert_eq!(notation.to_notation(), "d4-d5");
    }

    #[test]
    fn action_to_notation_move_left() {
        // White pawn moves from D4 to C4 (left)
        let original_state = State::new([Square::D4.to_mask(), Square::H8.to_mask()], 0);
        let action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::C4.to_mask());

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::D4);
        assert_eq!(notation.destination(), Square::C4);
        assert!(!notation.is_capture());
        assert_eq!(notation.to_notation(), "d4-c4");
    }

    #[test]
    fn action_to_notation_move_right() {
        // White pawn moves from D4 to E4 (right)
        let original_state = State::new([Square::D4.to_mask(), Square::H8.to_mask()], 0);
        let action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::E4.to_mask());

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::D4);
        assert_eq!(notation.destination(), Square::E4);
        assert!(!notation.is_capture());
        assert_eq!(notation.to_notation(), "d4-e4");
    }

    #[test]
    fn action_to_notation_pawn_promotion() {
        // White pawn moves from C7 to C8 and promotes
        let original_state = State::new([Square::C7.to_mask(), Square::H1.to_mask()], 0);
        let action = Action::new_move_as_pawn::<0>(Square::C7.to_mask(), Square::C8.to_mask());

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::C7);
        assert_eq!(notation.destination(), Square::C8);
        assert!(!notation.is_capture());
        assert!(notation.is_promotion());
        assert_eq!(notation.to_notation(), "c7-c8=K");
    }

    #[test]
    fn action_to_notation_black_pawn_promotion() {
        // Black pawn moves from D2 to D1 and promotes
        let original_state = State::new([Square::H8.to_mask(), Square::D2.to_mask()], 0);
        let action = Action::new_move_as_pawn::<1>(Square::D2.to_mask(), Square::D1.to_mask());

        let notation = action.to_detailed(Team::Black, &original_state);
        assert_eq!(notation.source(), Square::D2);
        assert_eq!(notation.destination(), Square::D1);
        assert!(!notation.is_capture());
        assert!(notation.is_promotion());
        assert_eq!(notation.to_notation(), "d2-d1=K");
    }

    #[test]
    fn action_to_notation_king_move() {
        // White king moves from D4 to D8
        let original_state = State::new(
            [Square::D4.to_mask(), Square::H1.to_mask()],
            Square::D4.to_mask(),
        );
        let action = Action::new_move_as_king::<0>(Square::D4.to_mask(), Square::D8.to_mask());

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::D4);
        assert_eq!(notation.destination(), Square::D8);
        assert!(!notation.is_capture());
        assert!(!notation.is_promotion()); // King doesn't promote
        assert_eq!(notation.to_notation(), "d4-d8");
    }

    #[test]
    fn action_to_notation_single_capture() {
        // White pawn at D4 captures black at D5, lands on D6
        let original_state = State::new([Square::D4.to_mask(), Square::D5.to_mask()], 0);
        let action = Action::new_capture_as_pawn::<0>(
            Square::D4.to_mask(),
            Square::D6.to_mask(),
            Square::D5.to_mask(),
            0,
        );

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::D4);
        assert_eq!(notation.destination(), Square::D6);
        assert!(notation.is_capture());
        assert!(!notation.is_promotion());
        assert_eq!(notation.to_notation(), "d4xd6");
    }

    #[test]
    fn action_to_notation_capture_with_promotion() {
        // White pawn at B6 captures black at B7, lands on B8 and promotes
        let original_state = State::new([Square::B6.to_mask(), Square::B7.to_mask()], 0);
        // First create the capture action
        let mut action = Action::new_capture_as_pawn::<0>(
            Square::B6.to_mask(),
            Square::B8.to_mask(),
            Square::B7.to_mask(),
            0,
        );
        // Manually add promotion (as done in generate_pawn_captures_at)
        action.delta.kings ^= Square::B8.to_mask();

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::B6);
        assert_eq!(notation.destination(), Square::B8);
        assert!(notation.is_capture());
        assert!(notation.is_promotion());
        assert_eq!(notation.to_notation(), "b6xb8=K");
    }

    #[test]
    fn action_to_notation_king_capture() {
        // White king at A4 captures black at D4, lands on H4
        let original_state = State::new(
            [Square::A4.to_mask(), Square::D4.to_mask()],
            Square::A4.to_mask(),
        );
        let action = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::H4.to_mask(),
            Square::D4.to_mask(),
            Square::A4.to_mask(),
        );

        let notation = action.to_detailed(Team::White, &original_state);
        assert_eq!(notation.source(), Square::A4);
        assert_eq!(notation.destination(), Square::H4);
        assert!(notation.is_capture());
        assert!(!notation.is_promotion());
        assert_eq!(notation.to_notation(), "a4xh4");
    }

    #[test]
    fn action_to_detailed_multi_capture_reconstructs_path() {
        // White pawn at A4 captures A5->A6, then B6->C6 (multi-capture)
        // The to_detailed method should reconstruct the intermediate landing A6
        let original_state = State::new(
            [
                Square::A4.to_mask(),
                Square::A5.to_mask() | Square::B6.to_mask(),
            ],
            0,
        );

        // Create first capture: A4 captures A5, lands A6
        let action1 = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            0,
        );

        // Create second capture: A6 captures B6, lands C6
        let action2 = Action::new_capture_as_pawn::<0>(
            Square::A6.to_mask(),
            Square::C6.to_mask(),
            Square::B6.to_mask(),
            0,
        );

        // Combine them
        let combined = action1.combine(&action2);

        let detailed = combined.to_detailed(Team::White, &original_state);
        // Multi-capture: source is A4, intermediate landing A6, final destination C6
        assert_eq!(detailed.source(), Square::A4);
        assert_eq!(detailed.destination(), Square::C6);
        assert!(detailed.is_capture());
        assert!(!detailed.is_promotion());
        // The path should include the intermediate landing square A6
        assert_eq!(detailed.path(), &[Square::A4, Square::A6, Square::C6]);
        assert_eq!(detailed.to_notation(), "a4xa6xc6");
    }

    #[test]
    fn action_source_helper() {
        let original_pieces = Square::D4.to_mask();
        let action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::D5.to_mask());

        assert_eq!(action.source(Team::White, original_pieces), Square::D4);
    }

    #[test]
    fn action_destination_helper() {
        let original_pieces = Square::D4.to_mask();
        let action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::D5.to_mask());

        assert_eq!(action.destination(Team::White, original_pieces), Square::D5);
    }

    #[test]
    fn action_is_capture_helper() {
        // Move - not a capture
        let move_action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::D5.to_mask());
        assert!(!move_action.is_capture(Team::White));

        // Capture
        let capture_action = Action::new_capture_as_pawn::<0>(
            Square::D4.to_mask(),
            Square::D6.to_mask(),
            Square::D5.to_mask(),
            0,
        );
        assert!(capture_action.is_capture(Team::White));
    }

    #[test]
    fn action_capture_count_helper() {
        // Single capture
        let single_capture = Action::new_capture_as_pawn::<0>(
            Square::D4.to_mask(),
            Square::D6.to_mask(),
            Square::D5.to_mask(),
            0,
        );
        assert_eq!(single_capture.capture_count(Team::White), 1);

        // Multi-capture (combined)
        let action1 = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::A6.to_mask(),
            Square::A5.to_mask(),
            0,
        );
        let action2 = Action::new_capture_as_pawn::<0>(
            Square::A6.to_mask(),
            Square::C6.to_mask(),
            Square::B6.to_mask(),
            0,
        );
        let combined = action1.combine(&action2);
        assert_eq!(combined.capture_count(Team::White), 2);
    }

    #[test]
    fn action_captured_pieces_helper() {
        let action = Action::new_capture_as_pawn::<0>(
            Square::D4.to_mask(),
            Square::D6.to_mask(),
            Square::D5.to_mask(),
            0,
        );
        assert_eq!(action.captured_pieces(Team::White), Square::D5.to_mask());
    }

    #[test]
    fn action_is_promotion_helper() {
        // Non-promotion move
        let original_state = State::new([Square::D4.to_mask(), Square::H8.to_mask()], 0);
        let move_action = Action::new_move_as_pawn::<0>(Square::D4.to_mask(), Square::D5.to_mask());
        assert!(!move_action.is_promotion(Team::White, &original_state));

        // Promotion move
        let promo_state = State::new([Square::C7.to_mask(), Square::H1.to_mask()], 0);
        let promo_action =
            Action::new_move_as_pawn::<0>(Square::C7.to_mask(), Square::C8.to_mask());
        assert!(promo_action.is_promotion(Team::White, &promo_state));

        // King move to promotion row (not a promotion, already a king)
        let king_state = State::new(
            [Square::C7.to_mask(), Square::H1.to_mask()],
            Square::C7.to_mask(), // C7 is already a king
        );
        let king_action = Action::new_move_as_king::<0>(Square::C7.to_mask(), Square::C8.to_mask());
        assert!(!king_action.is_promotion(Team::White, &king_state));
    }

    #[test]
    fn action_black_pawn_move() {
        // Black pawn moves from D5 to D4
        let original_state = State::new([Square::H1.to_mask(), Square::D5.to_mask()], 0);
        let action = Action::new_move_as_pawn::<1>(Square::D5.to_mask(), Square::D4.to_mask());

        let notation = action.to_detailed(Team::Black, &original_state);
        assert_eq!(notation.source(), Square::D5);
        assert_eq!(notation.destination(), Square::D4);
        assert!(!notation.is_capture());
        assert_eq!(notation.to_notation(), "d5-d4");
    }

    #[test]
    fn action_black_capture() {
        // Black pawn at D5 captures white at D4, lands on D3
        let original_state = State::new([Square::D4.to_mask(), Square::D5.to_mask()], 0);
        let action = Action::new_capture_as_pawn::<1>(
            Square::D5.to_mask(),
            Square::D3.to_mask(),
            Square::D4.to_mask(),
            0,
        );

        let notation = action.to_detailed(Team::Black, &original_state);
        assert_eq!(notation.source(), Square::D5);
        assert_eq!(notation.destination(), Square::D3);
        assert!(notation.is_capture());
        assert_eq!(notation.to_notation(), "d5xd3");
    }

    #[test]
    fn action_to_detailed_triple_capture_reconstructs_path() {
        // White pawn at D2 captures D3->D4, then D5->D6, then E6->F6
        // Path: D2 -> D4 -> D6 -> F6
        let original_state = State::new(
            [
                Square::D2.to_mask(),
                Square::D3.to_mask() | Square::D5.to_mask() | Square::E6.to_mask(),
            ],
            0,
        );

        // Create captures in sequence
        let action1 = Action::new_capture_as_pawn::<0>(
            Square::D2.to_mask(),
            Square::D4.to_mask(),
            Square::D3.to_mask(),
            0,
        );
        let action2 = Action::new_capture_as_pawn::<0>(
            Square::D4.to_mask(),
            Square::D6.to_mask(),
            Square::D5.to_mask(),
            0,
        );
        let action3 = Action::new_capture_as_pawn::<0>(
            Square::D6.to_mask(),
            Square::F6.to_mask(),
            Square::E6.to_mask(),
            0,
        );

        // Combine all three
        let combined = action1.combine(&action2).combine(&action3);

        let detailed = combined.to_detailed(Team::White, &original_state);
        assert_eq!(detailed.source(), Square::D2);
        assert_eq!(detailed.destination(), Square::F6);
        assert!(detailed.is_capture());
        assert_eq!(
            detailed.path(),
            &[Square::D2, Square::D4, Square::D6, Square::F6]
        );
        assert_eq!(detailed.to_notation(), "d2xd4xd6xf6");
    }

    #[test]
    fn action_to_detailed_king_multi_capture() {
        // White king at A4 captures D4->E4, then E7->E8
        // Path: A4 -> E4 -> E8
        let original_state = State::new(
            [
                Square::A4.to_mask(),
                Square::D4.to_mask() | Square::E7.to_mask(),
            ],
            Square::A4.to_mask(), // A4 is a king
        );

        // King capture: A4 captures D4, lands E4 (one square past)
        let action1 = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::E4.to_mask(),
            Square::D4.to_mask(),
            Square::A4.to_mask(),
        );
        // King capture: E4 captures E7, lands E8
        let action2 = Action::new_capture_as_king::<0>(
            Square::E4.to_mask(),
            Square::E8.to_mask(),
            Square::E7.to_mask(),
            Square::E4.to_mask(),
        );

        let combined = action1.combine(&action2);

        let detailed = combined.to_detailed(Team::White, &original_state);
        assert_eq!(detailed.source(), Square::A4);
        assert_eq!(detailed.destination(), Square::E8);
        assert!(detailed.is_capture());
        assert_eq!(detailed.path(), &[Square::A4, Square::E4, Square::E8]);
        assert_eq!(detailed.to_notation(), "a4xe4xe8");
    }

    // =====================================================================
    // Action::new simple API tests
    // =====================================================================

    #[test]
    fn action_new_white_pawn_move() {
        // White pawn moves from D3 to D4
        let action = Action::new(Team::White, Square::D3, Square::D4, &[], 0);

        // Verify it matches the low-level API
        let expected = Action::new_move_as_pawn::<0>(Square::D3.to_mask(), Square::D4.to_mask());
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_white_pawn_capture() {
        // White pawn at D4 captures black pawn at D5, landing on D6
        let action = Action::new(Team::White, Square::D4, Square::D6, &[Square::D5], 0);

        // Verify it matches the low-level API
        let expected = Action::new_capture_as_pawn::<0>(
            Square::D4.to_mask(),
            Square::D6.to_mask(),
            Square::D5.to_mask(),
            0,
        );
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_white_king_move() {
        // White king moves from D4 to D8
        let kings_mask = Square::D4.to_mask();
        let action = Action::new(Team::White, Square::D4, Square::D8, &[], kings_mask);

        // Verify it matches the low-level API
        let expected = Action::new_move_as_king::<0>(Square::D4.to_mask(), Square::D8.to_mask());
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_white_king_capture() {
        // White king at A4 captures black pawn at D4, landing on H4
        let kings_mask = Square::A4.to_mask();
        let action = Action::new(
            Team::White,
            Square::A4,
            Square::H4,
            &[Square::D4],
            kings_mask,
        );

        // Verify it matches the low-level API
        let expected = Action::new_capture_as_king::<0>(
            Square::A4.to_mask(),
            Square::H4.to_mask(),
            Square::D4.to_mask(),
            kings_mask,
        );
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_black_pawn_move() {
        // Black pawn moves from D6 to D5
        let action = Action::new(Team::Black, Square::D6, Square::D5, &[], 0);

        // Verify it matches the low-level API
        let expected = Action::new_move_as_pawn::<1>(Square::D6.to_mask(), Square::D5.to_mask());
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_black_pawn_capture() {
        // Black pawn at D5 captures white pawn at D4, landing on D3
        let action = Action::new(Team::Black, Square::D5, Square::D3, &[Square::D4], 0);

        // Verify it matches the low-level API
        let expected = Action::new_capture_as_pawn::<1>(
            Square::D5.to_mask(),
            Square::D3.to_mask(),
            Square::D4.to_mask(),
            0,
        );
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_black_king_move() {
        // Black king moves from D5 to D1
        let kings_mask = Square::D5.to_mask();
        let action = Action::new(Team::Black, Square::D5, Square::D1, &[], kings_mask);

        // Verify it matches the low-level API
        let expected = Action::new_move_as_king::<1>(Square::D5.to_mask(), Square::D1.to_mask());
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_black_king_capture() {
        // Black king at H4 captures white pawn at D4, landing on A4
        let kings_mask = Square::H4.to_mask();
        let action = Action::new(
            Team::Black,
            Square::H4,
            Square::A4,
            &[Square::D4],
            kings_mask,
        );

        // Verify it matches the low-level API
        let expected = Action::new_capture_as_king::<1>(
            Square::H4.to_mask(),
            Square::A4.to_mask(),
            Square::D4.to_mask(),
            kings_mask,
        );
        assert_eq!(action, expected);
    }

    #[test]
    fn action_new_multi_capture() {
        // White pawn captures multiple pieces
        let action = Action::new(
            Team::White,
            Square::A4,
            Square::C6,
            &[Square::A5, Square::B6],
            0,
        );

        // Verify capture mask includes both captured pieces
        let expected = Action::new_capture_as_pawn::<0>(
            Square::A4.to_mask(),
            Square::C6.to_mask(),
            Square::A5.to_mask() | Square::B6.to_mask(),
            0,
        );
        assert_eq!(action, expected);
    }

    #[test]
    fn action_is_empty() {
        // Empty sentinel action
        assert!(Action::EMPTY.is_empty());

        // Non-empty move action
        let move_action = Action::new(Team::White, Square::D3, Square::D4, &[], 0);
        assert!(!move_action.is_empty());

        // Non-empty capture action
        let capture_action = Action::new(Team::White, Square::D4, Square::D6, &[Square::D5], 0);
        assert!(!capture_action.is_empty());
    }
}
