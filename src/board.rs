//! Board representation and game logic for Turkish Draughts.
//!
//! This module defines the [`Board`] struct which combines the board state
//! with the current turn, providing the main interface for game logic.
//!
//! # Overview
//!
//! The `Board` struct is the primary type for interacting with the game:
//! - Query the current position and legal moves
//! - Apply moves to get new positions
//! - Check the game status (win/draw/in-progress)
//!
//! # Example
//!
//! ```rust
//! use kish::{Board, Team, GameStatus};
//!
//! // Create a new game
//! let mut board = Board::new_default();
//! assert_eq!(board.turn, Team::White);
//!
//! // Get legal moves
//! let actions = board.actions();
//! println!("Available moves: {}", actions.len());
//!
//! // Apply a move
//! if let Some(action) = actions.first() {
//!     board.apply_(action);
//!     board.swap_turn_();
//! }
//!
//! // Check game status
//! match board.status() {
//!     GameStatus::InProgress => println!("Game continues"),
//!     GameStatus::Draw => println!("Draw"),
//!     GameStatus::Won(team) => println!("{} wins!", team),
//! }
//! ```
//!
//! # Board Rotation
//!
//! The board can be rotated 180 degrees, which is useful for:
//! - Normalizing positions for transposition tables
//! - Viewing the board from the opponent's perspective
//! - Checking if the opponent is blocked
//!
//! ```rust
//! use kish::{Board, Team, Square};
//!
//! let board = Board::from_squares(
//!     Team::White,
//!     &[Square::A2],
//!     &[Square::H7],
//!     &[],
//! );
//!
//! let rotated = board.rotate();
//! // After rotation, the pieces swap positions (180° rotation)
//! ```

use std::fmt;

use super::{Action, GameStatus, Square, State, Team};
use crate::state::{
    MASK_COL_A, MASK_COL_B, MASK_COL_G, MASK_COL_H, MASK_ROW_1, MASK_ROW_2, MASK_ROW_7, MASK_ROW_8,
};

/// The main game board combining piece positions and current turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Board {
    /// The current team's turn.
    pub turn: Team,
    /// The state of the board.
    pub state: State,
}

impl Board {
    /// Creates a new board.
    #[must_use]
    pub const fn new(turn: Team, state: State) -> Self {
        Self { turn, state }
    }

    /// Creates a new board with default configuration.
    #[must_use]
    pub const fn new_default() -> Self {
        Self {
            turn: Team::White,
            state: State::default(),
        }
    }

    /// Creates a new board from the given squares.
    #[must_use]
    pub const fn from_squares(
        turn: Team,
        white_squares: &[Square],
        black_squares: &[Square],
        king_squares: &[Square],
    ) -> Self {
        let mut whites = 0u64;
        let mut i = 0;
        while i < white_squares.len() {
            let square = white_squares[i];
            whites |= square.to_mask();
            i += 1;
        }

        let mut blacks = 0u64;
        let mut i = 0;
        while i < black_squares.len() {
            let square = black_squares[i];
            blacks |= square.to_mask();
            i += 1;
        }

        let mut kings = 0u64;
        let mut i = 0;
        while i < king_squares.len() {
            let square = king_squares[i];
            kings |= square.to_mask();
            i += 1;
        }

        Self {
            turn,
            state: State {
                pieces: [whites, blacks],
                kings,
            },
        }
    }

    /// Returns the friendly pieces.
    #[inline(always)]
    #[must_use]
    pub const fn friendly_pieces(&self) -> u64 {
        self.state.pieces[self.turn as usize]
    }

    /// Returns the hostile pieces.
    #[inline(always)]
    #[must_use]
    pub const fn hostile_pieces(&self) -> u64 {
        self.state.pieces[self.turn.opponent() as usize]
    }

    /// Swaps the player's turn in-place.
    #[inline(always)]
    pub const fn swap_turn_(&mut self) {
        self.turn = self.turn.opponent();
    }

    /// Swaps the player's turn and returns the new board.
    #[inline]
    #[must_use]
    pub fn swap_turn(&self) -> Self {
        let mut new_board = *self;
        new_board.swap_turn_();
        new_board
    }

    /// Applies the given action to the board in-place.
    #[inline(always)]
    pub fn apply_(&mut self, action: &Action) {
        self.state.apply_(&action.delta);

        #[cfg(debug_assertions)]
        self.state.validate();
    }

    /// Applies the given action to the board and returns the new board.
    #[must_use]
    pub fn apply(&self, action: &Action) -> Self {
        let mut new_board = *self;
        new_board.apply_(action);
        new_board
    }

    /// Rotates the board by 180 degrees in-place.
    pub fn rotate_(&mut self) {
        self.state.rotate_();

        #[cfg(debug_assertions)]
        self.state.validate();
    }

    /// Rotates the board by 180 degrees and returns the new board.
    #[must_use]
    pub fn rotate(&self) -> Self {
        let mut new_board = *self;
        new_board.rotate_();
        new_board
    }

    /// Computes the status of the board.
    #[must_use]
    pub fn status(&self) -> GameStatus {
        let friendly_pieces = self.friendly_pieces();
        let hostile_pieces = self.hostile_pieces();

        // If friendlies have no pieces, then hostiles have won
        if friendly_pieces == 0 {
            return GameStatus::Won(self.turn.opponent());
        }

        // If hostiles have no pieces, then friendlies have won
        if hostile_pieces == 0 {
            return GameStatus::Won(self.turn);
        }

        // If friendlies have no actions, then hostiles have won. Being blocked only
        // loses on your own turn: the opponent being blocked while it is our turn is not
        // a result, since our (possibly forced) move may free them.
        if self.is_blocked() {
            return GameStatus::Won(self.turn.opponent());
        }

        // If both teams have a single piece, it is a draw - unless the side to move can
        // capture the last hostile piece right now, which wins (rule 8.1).
        if friendly_pieces.is_power_of_two()
            && hostile_pieces.is_power_of_two()
            && !self.can_capture_any()
        {
            return GameStatus::Draw;
        }

        // Otherwise, the game is in progress
        GameStatus::InProgress
    }

    /// Checks if the current player has no valid moves (is blocked).
    ///
    /// # Preconditions
    /// Caller must ensure both teams have pieces (checked by `status()`).
    const fn is_blocked(&self) -> bool {
        let friendly_pieces = self.friendly_pieces();
        let hostile_pieces = self.hostile_pieces();
        let friendly_kings = friendly_pieces & self.state.kings;
        let empty = self.state.empty();

        // Check if any piece can make a simple move (1 square)
        if self.can_any_piece_move(friendly_pieces, friendly_kings, empty) {
            return false;
        }

        // Check if any piece can make a simple capture (jumping 1 adjacent hostile),
        // including a king's backward capture
        if self.can_any_piece_capture_simple(friendly_pieces, friendly_kings, hostile_pieces, empty)
        {
            return false;
        }

        // Note: King flying captures are NOT checked here because they're redundant:
        // - Adjacent hostile with empty landing → found by can_any_piece_capture_simple
        // - Non-adjacent hostile (via ray) → requires empty adjacent squares → found by can_any_piece_move

        true
    }

    /// Whether the side to move has any capture (used only for the one-piece-each
    /// check, so the full move generator's cost is irrelevant).
    fn can_capture_any(&self) -> bool {
        self.actions()
            .first()
            .is_some_and(|action| action.is_capture(self.turn))
    }

    /// Checks if any piece can make a simple 1-square move.
    const fn can_any_piece_move(
        &self,
        friendly_pieces: u64,
        friendly_kings: u64,
        empty: u64,
    ) -> bool {
        // Left moves (all pieces)
        if ((friendly_pieces & !MASK_COL_A) >> 1) & empty != 0 {
            return true;
        }

        // Right moves (all pieces)
        if ((friendly_pieces & !MASK_COL_H) << 1) & empty != 0 {
            return true;
        }

        // Forward moves (team-dependent for pawns)
        let vertical_pawn_moves = match self.turn {
            Team::White => ((friendly_pieces & !MASK_ROW_8) << 8) & empty,
            Team::Black => ((friendly_pieces & !MASK_ROW_1) >> 8) & empty,
        };
        if vertical_pawn_moves != 0 {
            return true;
        }

        // Backward moves (kings only - opposite to team's forward direction)
        let backward_king_moves = match self.turn {
            Team::White => ((friendly_kings & !MASK_ROW_1) >> 8) & empty,
            Team::Black => ((friendly_kings & !MASK_ROW_8) << 8) & empty,
        };
        backward_king_moves != 0
    }

    /// Checks if any piece can make a simple capture (pawn-style: jump over 1 adjacent hostile).
    const fn can_any_piece_capture_simple(
        &self,
        friendly_pieces: u64,
        friendly_kings: u64,
        hostile_pieces: u64,
        empty: u64,
    ) -> bool {
        // Left capture: piece at col C+ can capture hostile 1 left, landing 2 left
        let left_captures =
            (((friendly_pieces & !(MASK_COL_A | MASK_COL_B)) >> 1) & hostile_pieces) >> 1 & empty;
        if left_captures != 0 {
            return true;
        }

        // Right capture: piece at col A-F can capture hostile 1 right, landing 2 right
        let right_captures =
            (((friendly_pieces & !(MASK_COL_G | MASK_COL_H)) << 1) & hostile_pieces) << 1 & empty;
        if right_captures != 0 {
            return true;
        }

        // Vertical capture (team-dependent direction)
        let vertical_captures = match self.turn {
            Team::White => {
                (((friendly_pieces & !(MASK_ROW_7 | MASK_ROW_8)) << 8) & hostile_pieces) << 8
                    & empty
            }
            Team::Black => {
                (((friendly_pieces & !(MASK_ROW_1 | MASK_ROW_2)) >> 8) & hostile_pieces) >> 8
                    & empty
            }
        };
        if vertical_captures != 0 {
            return true;
        }

        // Backward vertical capture (kings only). Missing this made a boxed king whose
        // only legal move is a backward capture count as blocked (a loss).
        let backward_king_captures = match self.turn {
            Team::White => {
                (((friendly_kings & !(MASK_ROW_1 | MASK_ROW_2)) >> 8) & hostile_pieces) >> 8 & empty
            }
            Team::Black => {
                (((friendly_kings & !(MASK_ROW_7 | MASK_ROW_8)) << 8) & hostile_pieces) << 8 & empty
            }
        };
        backward_king_captures != 0
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Status: {}\nTurn: {}\n{}",
            self.status(),
            self.turn,
            self.state
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{MASK_ROW_2, MASK_ROW_3, MASK_ROW_6};

    #[test]
    fn new() {
        let state = State::new(
            [Square::A4.to_mask(), Square::H5.to_mask()],
            Square::H5.to_mask(),
        );
        let board = Board::new(Team::Black, state);
        assert_eq!(board.turn, Team::Black);
        assert_eq!(board.state, state);
    }

    #[test]
    fn new_default() {
        let board = Board::new_default();
        assert_eq!(board.turn, Team::White);
        assert_eq!(board.state, State::default());
    }

    #[test]
    fn from_squares() {
        let friendly_squares = [Square::A1, Square::B2];
        let hostile_squares = [Square::C3, Square::D4];
        let king_squares = [Square::A1, Square::D4];
        let board = Board::from_squares(
            Team::Black,
            &friendly_squares,
            &hostile_squares,
            &king_squares,
        );

        assert_eq!(board.turn, Team::Black);
        assert_eq!(
            board.friendly_pieces(),
            Square::C3.to_mask() | Square::D4.to_mask()
        );
        assert_eq!(
            board.hostile_pieces(),
            Square::A1.to_mask() | Square::B2.to_mask()
        );
        assert_eq!(
            board.state.kings,
            Square::A1.to_mask() | Square::D4.to_mask()
        );
    }

    #[test]
    fn apply() {
        let board = Board::from_squares(
            Team::White,
            &[Square::A2, Square::B2],
            &[Square::A3, Square::D4],
            &[],
        );
        let action = Action::new(
            Team::White,
            Square::A2,
            Square::A4,
            &[Square::A3],
            board.state.kings,
        );
        let new_board = board.apply(&action);
        let expected =
            Board::from_squares(Team::White, &[Square::A4, Square::B2], &[Square::D4], &[]);
        assert_eq!(new_board, expected);

        // Test in-place
        let mut new_board_ = board;
        new_board_.apply_(&action);
        assert_eq!(new_board_, new_board);
    }

    #[test]
    fn rotate() {
        let board = Board::new(
            Team::White,
            State::new(
                [
                    MASK_ROW_2 | Square::B3.to_mask(),
                    MASK_ROW_6 | Square::F5.to_mask(),
                ],
                Square::B3.to_mask() | Square::F5.to_mask(),
            ),
        );

        let expected = Board::new(
            Team::White,
            State::new(
                [
                    MASK_ROW_3 | Square::C4.to_mask(),
                    MASK_ROW_7 | Square::G6.to_mask(),
                ],
                Square::G6.to_mask() | Square::C4.to_mask(),
            ),
        );
        let new_board = board.rotate();
        assert_eq!(new_board, expected);

        // Test in-place
        let mut new_board_ = board;
        new_board_.rotate_();
        assert_eq!(new_board_, new_board);
    }

    #[test]
    fn status_no_friendly_pieces() {
        let board = Board::from_squares(
            Team::White,
            &[],
            &[Square::A1, Square::B2, Square::C3, Square::D4],
            &[Square::A1],
        );
        assert_eq!(board.status(), GameStatus::Won(Team::Black));
    }

    #[test]
    fn status_no_hostile_pieces() {
        let board = Board::from_squares(
            Team::White,
            &[Square::A1, Square::B2, Square::C3, Square::D4],
            &[],
            &[Square::A1],
        );
        assert_eq!(board.status(), GameStatus::Won(Team::White));
    }

    #[test]
    fn status_draw() {
        let board = Board::from_squares(Team::White, &[Square::A1], &[Square::B2], &[Square::A1]);
        assert_eq!(board.status(), GameStatus::Draw);
    }

    #[test]
    fn status_in_progress() {
        let board = Board::from_squares(
            Team::White,
            &[Square::A2, Square::B2, Square::C3, Square::D4],
            &[Square::E5, Square::F6, Square::G7, Square::H8],
            &[Square::C3],
        );
        assert_eq!(board.status(), GameStatus::InProgress);
    }

    #[test]
    fn status_friendly_blocked() {
        let board = Board::from_squares(
            Team::White,
            &[Square::A2, Square::A3],
            &[
                Square::A4,
                Square::A5,
                Square::B2,
                Square::B3,
                Square::C2,
                Square::C3,
            ],
            &[],
        );
        assert_eq!(board.status(), GameStatus::Won(Team::Black));
    }

    #[test]
    fn status_hostile_blocked_is_not_a_result_on_our_turn() {
        let board = Board::from_squares(
            Team::White,
            &[
                Square::A2,
                Square::A3,
                Square::B4,
                Square::B5,
                Square::C4,
                Square::C5,
            ],
            &[Square::A4, Square::A5],
            &[],
        );
        // Black is blocked, but it is White's turn: being blocked only loses on your
        // own turn, and most White moves here free Black (e.g. c5-c6 allows a5xc5).
        assert_eq!(board.status(), GameStatus::InProgress);
        let freeing = board
            .actions()
            .into_iter()
            .filter(|a| !board.apply(a).swap_turn().actions().is_empty())
            .count();
        assert!(freeing > 0);
    }

    #[test]
    fn status_king_can_escape_via_flying_capture() {
        // King at A1 is surrounded but can make a flying capture
        // King can fly over empty squares and capture hostile at D1
        let board = Board::from_squares(
            Team::White,
            &[Square::A1],
            &[Square::A2, Square::B1, Square::D1],
            &[Square::A1], // A1 is a king
        );
        // King can capture D1 by flying right, so game is in progress
        assert_eq!(board.status(), GameStatus::InProgress);
    }

    #[test]
    fn status_king_blocked_no_flying_capture_possible() {
        // King at A1 is completely surrounded with no escape:
        // - B1, C1 block right movement and capture
        // - A2, A3 block up movement and capture
        let board = Board::from_squares(
            Team::White,
            &[Square::A1],
            &[Square::A2, Square::A3, Square::B1, Square::C1],
            &[Square::A1], // A1 is a king
        );
        // King cannot move or capture - blocked
        assert_eq!(board.status(), GameStatus::Won(Team::Black));
    }

    #[test]
    fn swap_turn_returns_new_board() {
        let board = Board::new_default();
        let swapped = board.swap_turn();
        assert_eq!(swapped.turn, Team::Black);
        assert_eq!(board.turn, Team::White); // Original unchanged
    }

    #[test]
    fn display_format() {
        let board = Board::from_squares(Team::White, &[Square::A1], &[Square::H8], &[]);
        let display = format!("{}", board);
        assert!(display.contains("Turn: White"));
        assert!(display.contains("Status:"));
    }

    #[test]
    fn default_board() {
        let board = Board::default();
        assert_eq!(board, Board::new_default());
    }

    #[test]
    fn status_black_pawn_blocked_can_capture_backward() {
        // Black pawn at A7 surrounded, but can capture down (backward for black)
        // This tests the vertical capture path for Black team
        let board = Board::from_squares(Team::Black, &[Square::A7], &[Square::A6, Square::B7], &[]);
        // Black pawn can capture A6 (moving down/backward)
        assert_eq!(board.status(), GameStatus::InProgress);
    }

    #[test]
    fn status_king_can_move_backward() {
        // White king at D4, surrounded except can move backward (down)
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::C4, Square::E4, Square::D5],
            &[Square::D4],
        );
        // King can move down to D3
        assert_eq!(board.status(), GameStatus::InProgress);
    }

    #[test]
    fn is_blocked_black_king_only_backward_move() {
        // Black king at A2, can move up (backward for Black) to A3
        // Left=edge, right=hostile, forward(down)=hostile
        let board = Board::from_squares(
            Team::Black,
            &[Square::A2],
            &[Square::A1, Square::B2],
            &[Square::A2],
        );
        assert_eq!(board.status(), GameStatus::InProgress);
    }

    #[test]
    fn is_blocked_black_vertical_capture_only() {
        // Black pawn at A3, can only capture forward (down to A1 over A2)
        // Left: edge. Right move: B3 hostile. Right capture: C3 hostile blocks landing.
        // Covers lines 320-321: Black vertical capture branch
        let board = Board::from_squares(
            Team::Black,
            &[Square::A3],
            &[Square::A2, Square::B3, Square::C3], // A2=capturable, B3=block right move, C3=block right capture
            &[],
        );
        assert_eq!(board.status(), GameStatus::InProgress);
    }

    #[test]
    fn is_blocked_left_capture_only() {
        // White pawn at C3, can only capture left (to A3 over B3)
        // Right: D3 hostile blocks move. E3 hostile blocks right capture landing.
        // Covers line 303: left capture return
        let board = Board::from_squares(
            Team::White,
            &[Square::C3],
            &[Square::B3, Square::C4, Square::D3, Square::E3], // B3=left capture, C4+D3=block moves, E3=block right capture
            &[],
        );
        assert_eq!(board.status(), GameStatus::InProgress);
    }
}
