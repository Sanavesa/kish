//! Full game management with history tracking for Turkish Draughts.
//!
//! This module provides the [`Game`] struct which wraps [`Board`] and adds:
//! - Position history tracking for threefold repetition detection
//! - Move counting for the insufficient progress rule
//!
//! # When to Use Game vs Board
//!
//! Use [`Board`] directly when:
//! - Performing AI search (alpha-beta, MCTS, etc.)
//! - Running perft tests
//! - You don't need draw detection beyond the basic 1v1 rule
//!
//! Use [`Game`] when:
//! - Playing a full game with proper draw detection
//! - You need to track game history
//! - You want to implement undo functionality
//!
//! # Example
//!
//! ```rust
//! use kish::{Game, GameStatus};
//!
//! let mut game = Game::new();
//!
//! // Play moves
//! let actions = game.actions();
//! if let Some(action) = actions.first() {
//!     game.make_move(action);
//! }
//!
//! // Check status (includes threefold repetition)
//! match game.status() {
//!     GameStatus::InProgress => println!("Game continues"),
//!     GameStatus::Draw => println!("Draw!"),
//!     GameStatus::Won(team) => println!("{} wins!", team),
//! }
//!
//! // Undo the last move
//! game.undo_move();
//! ```
//!
//! # Threefold Repetition (Rule 9.2)
//!
//! A draw is declared when the same position occurs three times with the same
//! player to move. The positions do not need to be consecutive.
//!
//! ```rust
//! use kish::{Game, Board, Team, Square, GameStatus};
//!
//! // Create a position where repetition can occur
//! let board = Board::from_squares(
//!     Team::White,
//!     &[Square::D4],
//!     &[Square::E5],
//!     &[Square::D4, Square::E5], // Both are kings
//! );
//! let mut game = Game::from_board(board);
//!
//! // The game tracks positions as moves are made
//! // Threefold repetition is automatically detected
//! ```
//!
//! # Insufficient Progress (Rule 9.4)
//!
//! A draw is declared after 150 consecutive plies (75 full moves) without any
//! capture. This prevents indefinitely prolonged endgames.

use rustc_hash::FxHashMap;

use crate::{Action, Board, GameStatus, Team};

/// Hash type for position lookup (single u64 for fast hashing).
type PositionHash = u64;

/// Number of half-moves (plies) without a capture before a draw is declared.
///
/// Per Rule 9.4, a draw occurs after this many consecutive plies without capture.
/// Set to 150 plies (75 full moves).
const INSUFFICIENT_PROGRESS_THRESHOLD: u16 = 150;

/// A full game with history tracking for proper draw detection.
///
/// This struct wraps a [`Board`] and maintains:
/// - A count of each position occurrence (for threefold repetition)
/// - A counter of half-moves since last capture (for insufficient progress)
/// - A history stack for undo functionality
///
/// # Memory Usage
///
/// The position history uses a `FxHashMap` which grows with unique positions.
/// For typical games, this is negligible. For very long games or analysis,
/// consider periodically clearing irrelevant history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Game {
    /// The current board state.
    board: Board,
    /// Count of each position occurrence for threefold repetition detection.
    position_counts: FxHashMap<PositionHash, u8>,
    /// Number of half-moves since last capture.
    /// A draw is declared at [`INSUFFICIENT_PROGRESS_THRESHOLD`] half-moves.
    halfmove_clock: u16,
    /// History stack for undo: (action, previous halfmove_clock, was_capture).
    history: Vec<(Action, u16, bool)>,
}

/// Typical game length for pre-allocation (most games end within 100 moves).
const TYPICAL_GAME_LENGTH: usize = 100;

impl Game {
    /// Creates a new game with the standard starting position.
    #[must_use]
    pub fn new() -> Self {
        let board = Board::new_default();
        let mut game = Self {
            board,
            position_counts: FxHashMap::with_capacity_and_hasher(
                TYPICAL_GAME_LENGTH,
                Default::default(),
            ),
            halfmove_clock: 0,
            history: Vec::with_capacity(TYPICAL_GAME_LENGTH),
        };
        game.record_position();
        game
    }

    /// Creates a game from an existing board position.
    ///
    /// The halfmove clock starts at 0 and the provided position is recorded
    /// as the first occurrence.
    #[must_use]
    pub fn from_board(board: Board) -> Self {
        let mut game = Self {
            board,
            position_counts: FxHashMap::with_capacity_and_hasher(
                TYPICAL_GAME_LENGTH,
                Default::default(),
            ),
            halfmove_clock: 0,
            history: Vec::with_capacity(TYPICAL_GAME_LENGTH),
        };
        game.record_position();
        game
    }

    /// Returns a reference to the current board.
    #[inline]
    #[must_use]
    pub const fn board(&self) -> &Board {
        &self.board
    }

    /// Returns the current team's turn.
    #[inline]
    #[must_use]
    pub const fn turn(&self) -> Team {
        self.board.turn
    }

    /// Returns the number of half-moves since the last capture.
    #[inline]
    #[must_use]
    pub const fn halfmove_clock(&self) -> u16 {
        self.halfmove_clock
    }

    /// Returns the number of moves in the game history.
    #[inline]
    #[must_use]
    pub fn move_count(&self) -> usize {
        self.history.len()
    }

    /// Returns all legal actions from the current position.
    #[inline]
    #[must_use]
    pub fn actions(&self) -> Vec<Action> {
        self.board.actions()
    }

    /// Computes the current game status, including draw conditions.
    ///
    /// This checks:
    /// 1. Win by capturing all pieces
    /// 2. Win by blocking the opponent
    /// 3. Draw by 1v1 (Rule 9.3)
    /// 4. Draw by threefold repetition (Rule 9.2)
    /// 5. Draw by insufficient progress (Rule 9.4) - 150 plies without capture
    #[must_use]
    pub fn status(&self) -> GameStatus {
        // First check basic status (win/loss/1v1 draw)
        let basic_status = self.board.status();
        if basic_status != GameStatus::InProgress {
            return basic_status;
        }

        // Check threefold repetition
        if self.is_threefold_repetition() {
            return GameStatus::Draw;
        }

        // Check insufficient progress
        if self.halfmove_clock >= INSUFFICIENT_PROGRESS_THRESHOLD {
            return GameStatus::Draw;
        }

        GameStatus::InProgress
    }

    /// Makes a move, updating the board and game history.
    ///
    /// This method:
    /// 1. Applies the action to the board
    /// 2. Swaps the turn
    /// 3. Records the new position for repetition detection
    /// 4. Updates the halfmove clock (resets on capture)
    /// 5. Pushes to history for undo support
    #[inline]
    pub fn make_move(&mut self, action: &Action) {
        let is_capture = self.is_capture_action(action);

        // Save current state for undo
        let prev_halfmove = self.halfmove_clock;

        // Apply the action
        self.board.apply_(action);
        self.board.swap_turn_();

        // Update halfmove clock (only captures reset)
        if is_capture {
            self.halfmove_clock = 0;
        } else {
            self.halfmove_clock += 1;
        }

        // Record position for repetition detection
        self.record_position();

        // Push to history
        self.history.push((*action, prev_halfmove, is_capture));
    }

    /// Undoes the last move, restoring the previous board state.
    ///
    /// Returns `true` if a move was undone, `false` if there was no move to undo.
    #[inline]
    pub fn undo_move(&mut self) -> bool {
        if let Some((action, prev_halfmove, _)) = self.history.pop() {
            // Decrement position count before undoing
            self.decrement_position_count();

            // Undo the turn swap
            self.board.swap_turn_();

            // Undo the action (XOR is self-inverse)
            self.board.apply_(&action);

            // Restore halfmove clock
            self.halfmove_clock = prev_halfmove;

            true
        } else {
            false
        }
    }

    /// Checks if the current position has occurred three times.
    #[inline]
    #[must_use]
    pub fn is_threefold_repetition(&self) -> bool {
        let hash = self.position_hash();
        self.position_counts.get(&hash).copied().unwrap_or(0) >= 3
    }

    /// Returns the number of times the current position has occurred.
    #[inline]
    #[must_use]
    pub fn position_occurrence_count(&self) -> u8 {
        let hash = self.position_hash();
        self.position_counts.get(&hash).copied().unwrap_or(0)
    }

    /// Clears the position history and resets the halfmove clock.
    ///
    /// This is useful when starting a new game from a position
    /// where prior history should not count.
    pub fn clear_history(&mut self) {
        self.position_counts.clear();
        self.history.clear();
        self.halfmove_clock = 0;
        self.record_position();
    }

    /// Perft (performance test) - counts leaf nodes at a given depth.
    ///
    /// This is similar to [`Board::perft`] but uses `make_move`/`undo_move`,
    /// which is useful for verifying that the Game's move/undo cycle is correct.
    ///
    /// # Note
    ///
    /// This is slower than `Board::perft` due to history tracking overhead.
    /// Use `Board::perft` for performance testing move generation.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kish::Game;
    ///
    /// let mut game = Game::new();
    /// let nodes = game.perft(3);
    /// println!("Positions at depth 3: {}", nodes);
    /// ```
    pub fn perft(&mut self, depth: u64) -> u64 {
        if depth == 0 {
            return 1;
        }

        let actions = self.actions();
        if actions.is_empty() {
            return 1; // Terminal node counts as 1
        }

        let mut nodes = 0u64;
        for action in &actions {
            self.make_move(action);
            nodes += self.perft(depth - 1);
            self.undo_move();
        }
        nodes
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Computes a hash for the current position (state + turn).
    #[inline]
    fn position_hash(&self) -> PositionHash {
        // XOR mixing with golden ratio constants for good distribution
        let mut hash = self.board.state.pieces[0];
        hash ^= self.board.state.pieces[1].wrapping_mul(0x9e37_79b9_7f4a_7c15);
        hash ^= self.board.state.kings.wrapping_mul(0x517c_c1b7_2722_0a95);
        hash ^= (self.board.turn.to_usize() as u64).wrapping_mul(0x2545_f491_4f6c_dd1d);
        hash
    }

    /// Records the current position in the occurrence map.
    #[inline]
    fn record_position(&mut self) {
        let hash = self.position_hash();
        *self.position_counts.entry(hash).or_insert(0) += 1;
    }

    /// Decrements the position count for the current position.
    #[inline]
    fn decrement_position_count(&mut self) {
        let hash = self.position_hash();
        if let Some(count) = self.position_counts.get_mut(&hash) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                self.position_counts.remove(&hash);
            }
        }
    }

    /// Checks if an action is a capture (removes opponent pieces).
    #[inline]
    fn is_capture_action(&self, action: &Action) -> bool {
        let opponent_index = self.board.turn.opponent().to_usize();
        action.delta.pieces[opponent_index] != 0
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

impl From<Board> for Game {
    fn from(board: Board) -> Self {
        Self::from_board(board)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Square;

    #[test]
    fn new_game_starts_at_default_position() {
        let game = Game::new();
        assert_eq!(game.board, Board::new_default());
        assert_eq!(game.turn(), Team::White);
        assert_eq!(game.halfmove_clock(), 0);
        assert_eq!(game.move_count(), 0);
    }

    #[test]
    fn from_board_preserves_position() {
        let board = Board::from_squares(
            Team::Black,
            &[Square::A2, Square::B2],
            &[Square::G7, Square::H7],
            &[],
        );
        let game = Game::from_board(board);
        assert_eq!(game.board, board);
        assert_eq!(game.turn(), Team::Black);
    }

    #[test]
    fn make_move_updates_board() {
        let mut game = Game::new();
        let actions = game.actions();
        assert!(!actions.is_empty());

        let action = actions[0];
        game.make_move(&action);

        // Turn should have swapped
        assert_eq!(game.turn(), Team::Black);
        assert_eq!(game.move_count(), 1);
    }

    #[test]
    fn undo_move_restores_board() {
        let mut game = Game::new();
        let original_board = game.board;

        let actions = game.actions();
        game.make_move(&actions[0]);
        assert_ne!(game.board, original_board);

        game.undo_move();
        assert_eq!(game.board, original_board);
        assert_eq!(game.move_count(), 0);
    }

    #[test]
    fn halfmove_clock_increments_for_king_moves() {
        // Create a position with only kings
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::E6],
            &[Square::D4, Square::E6], // Both are kings
        );
        let mut game = Game::from_board(board);

        // Make a king move (should increment clock)
        let actions = game.actions();
        game.make_move(&actions[0]);
        assert_eq!(game.halfmove_clock(), 1);
    }

    #[test]
    fn halfmove_clock_resets_on_capture() {
        // Create a position where capture is forced - use a pawn, not a king
        // Pawn captures are simpler (single landing square)
        let board = Board::from_squares(
            Team::White,
            &[Square::D4],
            &[Square::D5, Square::H8], // Need second piece to avoid 1v1 draw
            &[],                       // No kings
        );
        let mut game = Game::from_board(board);

        // Set clock to non-zero
        game.halfmove_clock = 10;

        // Make the capture (should reset clock)
        let actions = game.actions();
        assert_eq!(actions.len(), 1); // Only one capture possible (pawn lands on D6)
        game.make_move(&actions[0]);
        assert_eq!(game.halfmove_clock(), 0);
    }

    #[test]
    fn threefold_repetition_detection() {
        // Create a position with two kings each, far enough apart to avoid 1v1 draw.
        // Kings at A1/B1 for white, H7/H8 for black.
        // We'll shuffle A1<->A2 and H8<->G8 to create repetitions.
        let board = Board::from_squares(
            Team::White,
            &[Square::A1, Square::B1],
            &[Square::H7, Square::H8],
            &[Square::A1, Square::B1, Square::H7, Square::H8],
        );
        let mut game = Game::from_board(board);

        // Initial position is recorded once
        assert_eq!(game.position_occurrence_count(), 1);
        assert!(!game.is_threefold_repetition());

        // Find moves for the shuffle pattern.
        // White: A1 -> A2, Black: H8 -> G8, White: A2 -> A1, Black: G8 -> H8
        // This returns to the starting position.

        // Helper to find a specific move
        let find_move = |g: &Game, from: Square, to: Square| -> Action {
            g.actions()
                .into_iter()
                .find(|a| {
                    let from_mask = from.to_mask();
                    let to_mask = to.to_mask();
                    let friendly_idx = g.turn().to_usize();
                    let delta = a.delta.pieces[friendly_idx];
                    (delta & from_mask) != 0 && (delta & to_mask) != 0
                })
                .expect("Move not found")
        };

        // Round 1: back to start
        game.make_move(&find_move(&game, Square::A1, Square::A2)); // White A1->A2
        game.make_move(&find_move(&game, Square::H8, Square::G8)); // Black H8->G8
        game.make_move(&find_move(&game, Square::A2, Square::A1)); // White A2->A1
        game.make_move(&find_move(&game, Square::G8, Square::H8)); // Black G8->H8
        assert_eq!(game.position_occurrence_count(), 2);
        assert!(!game.is_threefold_repetition());

        // Round 2: back to start again
        game.make_move(&find_move(&game, Square::A1, Square::A2));
        game.make_move(&find_move(&game, Square::H8, Square::G8));
        game.make_move(&find_move(&game, Square::A2, Square::A1));
        game.make_move(&find_move(&game, Square::G8, Square::H8));
        assert_eq!(game.position_occurrence_count(), 3);
        assert!(game.is_threefold_repetition());
        assert_eq!(game.status(), GameStatus::Draw);
    }

    #[test]
    fn insufficient_progress_draw() {
        // Need more than 1 piece each to avoid 1v1 draw rule
        let board = Board::from_squares(
            Team::White,
            &[Square::D4, Square::A1],
            &[Square::E6, Square::H8],
            &[Square::D4, Square::A1, Square::E6, Square::H8], // All are kings
        );
        let mut game = Game::from_board(board);

        // Set clock to threshold - 1 (one short of draw)
        game.halfmove_clock = INSUFFICIENT_PROGRESS_THRESHOLD - 1;
        assert_eq!(game.status(), GameStatus::InProgress);

        // Set clock to threshold (draw)
        game.halfmove_clock = INSUFFICIENT_PROGRESS_THRESHOLD;
        assert_eq!(game.status(), GameStatus::Draw);
    }

    #[test]
    fn status_checks_all_conditions() {
        // Test 1v1 draw takes priority
        let board = Board::from_squares(Team::White, &[Square::A1], &[Square::H8], &[]);
        let game = Game::from_board(board);
        assert_eq!(game.status(), GameStatus::Draw);
    }

    #[test]
    fn clear_history_resets_everything() {
        let mut game = Game::new();
        let actions = game.actions();
        game.make_move(&actions[0]);
        game.halfmove_clock = 50;

        game.clear_history();

        assert_eq!(game.halfmove_clock(), 0);
        assert_eq!(game.move_count(), 0);
        assert_eq!(game.position_occurrence_count(), 1);
    }

    #[test]
    fn position_count_decrements_on_undo() {
        let mut game = Game::new();
        let initial_count = game.position_occurrence_count();

        let actions = game.actions();
        game.make_move(&actions[0]);

        // Verify undo decrements counts
        game.undo_move();
        assert_eq!(game.position_occurrence_count(), initial_count);
    }

    #[test]
    fn from_board_trait_impl() {
        let board = Board::from_squares(Team::Black, &[Square::C3], &[Square::F6], &[Square::C3]);
        let game: Game = board.into();
        assert_eq!(game.turn(), Team::Black);
        assert_eq!(game.halfmove_clock(), 0);
        assert_eq!(game.position_occurrence_count(), 1);
    }

    #[test]
    fn undo_move_returns_false_when_empty() {
        let mut game = Game::new();
        assert!(!game.undo_move());
    }

    #[test]
    fn default_trait_impl() {
        let game1 = Game::default();
        let game2 = Game::new();
        assert_eq!(game1, game2);
    }

    #[test]
    fn halfmove_clock_does_not_reset_on_pawn_move() {
        // Create position with a pawn that can move (no capture)
        let board = Board::from_squares(
            Team::White,
            &[Square::D4, Square::A1], // D4 is a pawn, A1 is a king
            &[Square::H7, Square::H8],
            &[Square::A1, Square::H7, Square::H8], // Only A1, H7, H8 are kings
        );
        let mut game = Game::from_board(board);
        game.halfmove_clock = 10;

        // Find the pawn move (D4 -> D5)
        let pawn_move = game
            .actions()
            .into_iter()
            .find(|a| {
                let d4 = Square::D4.to_mask();
                let d5 = Square::D5.to_mask();
                (a.delta.pieces[0] & d4) != 0 && (a.delta.pieces[0] & d5) != 0
            })
            .expect("Pawn move not found");

        game.make_move(&pawn_move);
        // Pawn moves do NOT reset clock (only captures do per Rule 9.4)
        assert_eq!(game.halfmove_clock(), 11);
    }

    #[test]
    fn insufficient_progress_threshold_is_configurable() {
        // Verify the const is used correctly
        assert_eq!(INSUFFICIENT_PROGRESS_THRESHOLD, 150);
    }

    #[test]
    fn perft_matches_board_perft() {
        // Verify Game::perft produces same results as Board::perft
        let board = Board::new_default();
        let mut game = Game::new();

        for depth in 0..=5 {
            let board_result = board.perft(depth);
            let game_result = game.perft(depth);
            assert_eq!(
                board_result, game_result,
                "Game::perft({}) = {} != Board::perft({}) = {}",
                depth, game_result, depth, board_result
            );
        }
    }

    #[test]
    fn perft_state_unchanged_after_call() {
        // Verify the game state is unchanged after perft
        let mut game = Game::new();
        let original_board = game.board;
        let original_count = game.move_count();

        game.perft(3);

        assert_eq!(game.board, original_board);
        assert_eq!(game.move_count(), original_count);
    }
}
