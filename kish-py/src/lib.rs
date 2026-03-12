//! Python bindings for the kish Turkish Draughts (Dama) engine.
//!
//! This module provides high-performance Python bindings for the kish library,
//! designed for both **UI applications** and **machine learning**.
//!
//! # Types
//!
//! - [`Team`]: Player enum (`White`, `Black`)
//! - [`Square`]: Board square enum (`A1` through `H8`)
//! - [`GameStatus`]: Game state with query methods
//! - [`Action`]: Move with notation and bitboard access
//! - [`Board`]: Immutable game board
//! - [`Game`]: Mutable game with history tracking
//!
//! # Design Philosophy
//!
//! - **UI-friendly**: Human-readable notation, square lists, intuitive methods
//! - **ML-friendly**: Fast bitboard access, numpy-compatible arrays, action deltas
//! - **Immutable Board**: `apply()` returns new board (functional style)
//! - **Mutable Game**: `make_move()` mutates state (imperative style with undo)

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// Use explicit path to avoid collision with the Python module name
use ::kish as kish_core;

// ============================================================================
// Team
// ============================================================================

/// A player in the game.
///
/// Turkish Draughts is played between two players: White and Black.
/// White always moves first in the standard starting position.
///
/// # Example
/// ```python
/// import kish
///
/// team = kish.Team.White
/// opponent = team.opponent()  # Team.Black
/// ```
#[pyclass(eq, eq_int, frozen, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Team {
    /// The white player (moves first).
    White = 0,
    /// The black player.
    Black = 1,
}

#[pymethods]
impl Team {
    /// Returns the opponent team.
    #[must_use]
    fn opponent(&self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::White => "Team.White",
            Self::Black => "Team.Black",
        }
    }

    fn __str__(&self) -> &'static str {
        match self {
            Self::White => "White",
            Self::Black => "Black",
        }
    }
}

impl From<kish_core::Team> for Team {
    fn from(team: kish_core::Team) -> Self {
        match team {
            kish_core::Team::White => Self::White,
            kish_core::Team::Black => Self::Black,
        }
    }
}

impl From<Team> for kish_core::Team {
    fn from(team: Team) -> Self {
        match team {
            Team::White => Self::White,
            Team::Black => Self::Black,
        }
    }
}

// ============================================================================
// Square
// ============================================================================

/// A square on the 8x8 board.
///
/// Squares are identified using algebraic notation (A1-H8).
/// The board is indexed with A1 at bottom-left (bit 0) and H8 at top-right (bit 63).
///
/// # Bitboard Representation
/// Each square maps to a single bit in a 64-bit integer:
/// - Bit index = row * 8 + column
/// - Row 0 = rank 1, Row 7 = rank 8
/// - Column 0 = file A, Column 7 = file H
///
/// # Example
/// ```python
/// import kish
///
/// # Create from notation
/// sq = kish.Square.from_notation("d4")
///
/// # Create from indices
/// sq = kish.Square.from_row_col(3, 3)  # D4
///
/// # Query properties
/// print(sq.row())       # 3
/// print(sq.col())       # 3
/// print(sq.notation())  # "D4"
///
/// # Bitboard operations
/// mask = sq.to_mask()   # 1 << 27
/// sq2 = kish.Square.from_mask(mask)
///
/// # Distance for heuristics
/// dist = kish.Square.D4.manhattan(kish.Square.H8)  # 8
/// ```
#[pyclass(eq, eq_int, frozen, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Square {
    A1 = 0,
    B1 = 1,
    C1 = 2,
    D1 = 3,
    E1 = 4,
    F1 = 5,
    G1 = 6,
    H1 = 7,
    A2 = 8,
    B2 = 9,
    C2 = 10,
    D2 = 11,
    E2 = 12,
    F2 = 13,
    G2 = 14,
    H2 = 15,
    A3 = 16,
    B3 = 17,
    C3 = 18,
    D3 = 19,
    E3 = 20,
    F3 = 21,
    G3 = 22,
    H3 = 23,
    A4 = 24,
    B4 = 25,
    C4 = 26,
    D4 = 27,
    E4 = 28,
    F4 = 29,
    G4 = 30,
    H4 = 31,
    A5 = 32,
    B5 = 33,
    C5 = 34,
    D5 = 35,
    E5 = 36,
    F5 = 37,
    G5 = 38,
    H5 = 39,
    A6 = 40,
    B6 = 41,
    C6 = 42,
    D6 = 43,
    E6 = 44,
    F6 = 45,
    G6 = 46,
    H6 = 47,
    A7 = 48,
    B7 = 49,
    C7 = 50,
    D7 = 51,
    E7 = 52,
    F7 = 53,
    G7 = 54,
    H7 = 55,
    A8 = 56,
    B8 = 57,
    C8 = 58,
    D8 = 59,
    E8 = 60,
    F8 = 61,
    G8 = 62,
    H8 = 63,
}

#[pymethods]
impl Square {
    /// Creates a square from algebraic notation (e.g., "d4", "H8").
    #[staticmethod]
    fn from_notation(notation: &str) -> PyResult<Self> {
        let inner: kish_core::Square = notation
            .parse()
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;
        Ok(inner.into())
    }

    /// Creates a square from row and column indices (0-7 each).
    #[staticmethod]
    fn from_row_col(row: u8, col: u8) -> PyResult<Self> {
        if row >= 8 || col >= 8 {
            return Err(PyValueError::new_err("row and col must be in range 0-7"));
        }
        let index = row * 8 + col;
        // SAFETY: we validated the range
        Ok(unsafe { kish_core::Square::from_u8(index) }.into())
    }

    /// Returns the row (rank) of this square (0-7, where 0 is rank 1).
    #[must_use]
    fn row(&self) -> u8 {
        kish_core::Square::from(*self).row()
    }

    /// Returns the column (file) of this square (0-7, where 0 is file A).
    #[must_use]
    fn col(&self) -> u8 {
        kish_core::Square::from(*self).column()
    }

    /// Returns the algebraic notation for this square (e.g., "D4").
    #[must_use]
    fn notation(&self) -> String {
        kish_core::Square::from(*self).to_string()
    }

    /// Returns the Manhattan distance to another square.
    ///
    /// Useful for heuristics (e.g., distance to promotion row).
    #[must_use]
    fn manhattan(&self, other: Square) -> u8 {
        kish_core::Square::from(*self).manhattan(other.into())
    }

    /// Returns this square as a bitboard mask (u64 with one bit set).
    #[must_use]
    fn to_mask(&self) -> u64 {
        kish_core::Square::from(*self).to_mask()
    }

    /// Creates a square from a bitboard mask.
    ///
    /// The mask must have exactly one bit set.
    #[staticmethod]
    fn from_mask(mask: u64) -> PyResult<Self> {
        if mask == 0 || (mask & (mask - 1)) != 0 {
            return Err(PyValueError::new_err("mask must have exactly one bit set"));
        }
        // SAFETY: we validated the mask has exactly one bit set
        Ok(unsafe { kish_core::Square::from_mask(mask) }.into())
    }

    fn __repr__(&self) -> String {
        format!("Square.{}", self.notation())
    }

    fn __str__(&self) -> String {
        self.notation()
    }
}

impl From<kish_core::Square> for Square {
    fn from(sq: kish_core::Square) -> Self {
        // SAFETY: both enums have the same repr(u8) with values 0-63
        unsafe { std::mem::transmute(sq) }
    }
}

impl From<Square> for kish_core::Square {
    fn from(sq: Square) -> Self {
        // SAFETY: both enums have the same repr(u8) with values 0-63
        unsafe { std::mem::transmute(sq) }
    }
}

// ============================================================================
// GameStatus
// ============================================================================

/// The current status of a game.
///
/// A game can be in one of three states:
/// - **In Progress**: The game is ongoing
/// - **Draw**: The game ended in a draw (1v1, threefold repetition, or 150-move rule)
/// - **Won**: A player has won (captured all pieces or blocked opponent)
///
/// # Example
/// ```python
/// import kish
///
/// board = kish.Board()
/// status = board.status()
///
/// if status.is_in_progress():
///     print("Game continues")
/// elif status.is_draw():
///     print("Draw!")
/// elif status.is_won():
///     print(f"Winner: {status.winner()}")
/// ```
#[pyclass(eq, frozen)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct GameStatus {
    inner: kish_core::GameStatus,
}

#[pymethods]
impl GameStatus {
    /// Returns true if the game is still in progress.
    #[must_use]
    fn is_in_progress(&self) -> bool {
        matches!(self.inner, kish_core::GameStatus::InProgress)
    }

    /// Returns true if the game ended in a draw.
    #[must_use]
    fn is_draw(&self) -> bool {
        matches!(self.inner, kish_core::GameStatus::Draw)
    }

    /// Returns true if the game has been won.
    #[must_use]
    fn is_won(&self) -> bool {
        matches!(self.inner, kish_core::GameStatus::Won(_))
    }

    /// Returns true if the game is over (draw or won).
    #[must_use]
    fn is_over(&self) -> bool {
        self.inner.is_over()
    }

    /// Returns the winning team if the game was won, otherwise None.
    #[must_use]
    fn winner(&self) -> Option<Team> {
        match self.inner {
            kish_core::GameStatus::Won(team) => Some(team.into()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            kish_core::GameStatus::InProgress => "GameStatus.InProgress".to_string(),
            kish_core::GameStatus::Draw => "GameStatus.Draw".to_string(),
            kish_core::GameStatus::Won(team) => {
                format!("GameStatus.Won({})", Team::from(team).__repr__())
            }
        }
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

impl From<kish_core::GameStatus> for GameStatus {
    fn from(status: kish_core::GameStatus) -> Self {
        Self { inner: status }
    }
}

// ============================================================================
// Action
// ============================================================================

/// A move in the game.
///
/// Actions represent legal moves and provide both UI-friendly methods
/// (notation, square lists) and ML-friendly methods (bitboard deltas).
///
/// # Notation Format
/// - Simple move: `d3-d4`
/// - Capture: `d4xd6`
/// - Multi-capture: `d4xd6xf6`
/// - Promotion: `d7-d8=K`
///
/// # Example
/// ```python
/// import kish
///
/// board = kish.Board()
/// actions = board.actions()
///
/// for action in actions:
///     # UI methods
///     print(action.notation())           # "a3-a4"
///     print(action.source())             # Square.A3
///     print(action.destination())        # Square.A4
///     print(action.is_capture())         # False
///
///     # ML methods
///     w_delta, b_delta, k_delta = action.delta()
///     captured_bb = action.captured_bitboard()
/// ```
#[pyclass(frozen)]
#[derive(Clone)]
pub struct Action {
    inner: kish_core::Action,
    /// Cached detailed representation for notation
    detailed: kish_core::ActionPath,
    /// The team that made this move (needed for captured_pieces)
    team: kish_core::Team,
}

#[pymethods]
impl Action {
    /// Returns the source square of this move.
    #[must_use]
    fn source(&self) -> Square {
        self.detailed.source().into()
    }

    /// Returns the destination square of this move.
    #[must_use]
    fn destination(&self) -> Square {
        self.detailed.destination().into()
    }

    /// Returns true if this move is a capture.
    #[must_use]
    fn is_capture(&self) -> bool {
        self.detailed.is_capture()
    }

    /// Returns true if this move results in promotion to king.
    #[must_use]
    fn is_promotion(&self) -> bool {
        self.detailed.is_promotion()
    }

    /// Returns the number of pieces captured (0 for non-captures).
    #[must_use]
    fn capture_count(&self) -> u8 {
        if self.detailed.is_capture() {
            (self.detailed.path_len() - 1) as u8
        } else {
            0
        }
    }

    /// Returns the full path of squares visited during this move.
    #[must_use]
    fn path(&self) -> Vec<Square> {
        self.detailed.path().iter().map(|&sq| sq.into()).collect()
    }

    /// Returns the algebraic notation for this move (e.g., "d3-d4", "d4xd6").
    #[must_use]
    fn notation(&self) -> String {
        self.detailed.to_notation()
    }

    // =========================================================================
    // Bitboard access for ML
    // =========================================================================

    /// Returns the captured pieces as a list of squares.
    ///
    /// Empty list for non-captures.
    #[must_use]
    fn captured_pieces(&self) -> Vec<Square> {
        Board::mask_to_squares(self.inner.captured_pieces(self.team))
    }

    /// Returns the captured pieces as a bitboard (u64).
    ///
    /// Zero for non-captures.
    #[must_use]
    fn captured_bitboard(&self) -> u64 {
        self.inner.captured_pieces(self.team)
    }

    /// Returns the action delta as a tuple of bitboards.
    ///
    /// Returns (white_delta, black_delta, kings_delta).
    /// XOR these with the board state to apply/unapply the action.
    #[must_use]
    fn delta(&self) -> (u64, u64, u64) {
        (
            self.inner.delta.pieces[0],
            self.inner.delta.pieces[1],
            self.inner.delta.kings,
        )
    }

    /// Returns the action delta as an array [white_delta, black_delta, kings_delta].
    ///
    /// Convenient for numpy: `np.array(action.delta_array(), dtype=np.uint64)`
    #[must_use]
    fn delta_array(&self) -> [u64; 3] {
        [
            self.inner.delta.pieces[0],
            self.inner.delta.pieces[1],
            self.inner.delta.kings,
        ]
    }

    fn __repr__(&self) -> String {
        format!("Action('{}')", self.notation())
    }

    fn __str__(&self) -> String {
        self.notation()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

// ============================================================================
// Board
// ============================================================================

/// An immutable game board with piece positions and current turn.
///
/// `Board` is the core type for game state. It uses **immutable semantics**:
/// calling `apply()` returns a new board rather than modifying the original.
/// This is ideal for game tree search and functional programming.
///
/// For mutable game state with history tracking, use [`Game`] instead.
///
/// # Creating Boards
/// ```python
/// import kish
///
/// # Standard starting position
/// board = kish.Board()
///
/// # Custom position from squares
/// board = kish.Board.from_squares(
///     turn=kish.Team.White,
///     white_squares=[kish.Square.D4],
///     black_squares=[kish.Square.E5],
///     king_squares=[kish.Square.D4],
/// )
///
/// # Custom position from bitboards (for ML)
/// board = kish.Board.from_bitboards(turn=0, white=0xFF00, black=0xFF000000, kings=0)
/// ```
///
/// # Playing Moves
/// ```python
/// actions = board.actions()
/// new_board = board.apply(actions[0])  # Returns new board
/// ```
///
/// # ML Access
/// ```python
/// white, black, kings, turn = board.bitboards()
/// arr = board.to_array()  # For numpy
/// ```
#[pyclass]
#[derive(Clone)]
pub struct Board {
    inner: kish_core::Board,
}

#[pymethods]
impl Board {
    /// Creates a new board with the standard starting position.
    #[new]
    fn new() -> Self {
        Self {
            inner: kish_core::Board::new_default(),
        }
    }

    /// Creates a board from custom piece positions.
    ///
    /// Args:
    ///     turn: The team to move.
    ///     white_squares: List of squares with white pieces.
    ///     black_squares: List of squares with black pieces.
    ///     king_squares: List of squares that are kings (must be subset of pieces).
    #[staticmethod]
    fn from_squares(
        turn: Team,
        white_squares: Vec<Square>,
        black_squares: Vec<Square>,
        king_squares: Vec<Square>,
    ) -> Self {
        let whites: Vec<kish_core::Square> = white_squares.into_iter().map(Into::into).collect();
        let blacks: Vec<kish_core::Square> = black_squares.into_iter().map(Into::into).collect();
        let kings: Vec<kish_core::Square> = king_squares.into_iter().map(Into::into).collect();

        Self {
            inner: kish_core::Board::from_squares(turn.into(), &whites, &blacks, &kings),
        }
    }

    /// Returns the current team's turn.
    #[getter]
    fn turn(&self) -> Team {
        self.inner.turn.into()
    }

    /// Returns all legal actions from the current position.
    #[must_use]
    fn actions(&self) -> Vec<Action> {
        let team = self.inner.turn;
        self.inner
            .actions()
            .into_iter()
            .map(|action| {
                let detailed = action.to_detailed(team, &self.inner.state);
                Action {
                    inner: action,
                    detailed,
                    team,
                }
            })
            .collect()
    }

    /// Applies an action and returns a new board with the turn swapped.
    #[must_use]
    fn apply(&self, action: &Action) -> Self {
        let mut new_board = self.inner.apply(&action.inner);
        new_board.swap_turn_();
        Self { inner: new_board }
    }

    /// Returns the current game status.
    #[must_use]
    fn status(&self) -> GameStatus {
        self.inner.status().into()
    }

    /// Returns the squares occupied by white pieces.
    #[must_use]
    fn white_pieces(&self) -> Vec<Square> {
        Self::mask_to_squares(self.inner.state.pieces[0])
    }

    /// Returns the squares occupied by black pieces.
    #[must_use]
    fn black_pieces(&self) -> Vec<Square> {
        Self::mask_to_squares(self.inner.state.pieces[1])
    }

    /// Returns the squares occupied by kings.
    #[must_use]
    fn kings(&self) -> Vec<Square> {
        Self::mask_to_squares(self.inner.state.kings)
    }

    /// Returns the squares occupied by the current player's pieces.
    #[must_use]
    fn friendly_pieces(&self) -> Vec<Square> {
        Self::mask_to_squares(self.inner.friendly_pieces())
    }

    /// Returns the squares occupied by the opponent's pieces.
    #[must_use]
    fn hostile_pieces(&self) -> Vec<Square> {
        Self::mask_to_squares(self.inner.hostile_pieces())
    }

    /// Returns a rotated copy of the board (180 degrees).
    #[must_use]
    fn rotate(&self) -> Self {
        Self {
            inner: self.inner.rotate(),
        }
    }

    /// Runs a perft (performance test) at the given depth.
    ///
    /// Returns the number of leaf nodes (positions) at that depth.
    #[must_use]
    fn perft(&self, depth: u64) -> u64 {
        self.inner.perft(depth)
    }

    // =========================================================================
    // Bitboard access for ML
    // =========================================================================

    /// Returns the white pieces bitboard as a u64.
    ///
    /// Each bit represents a square (bit 0 = A1, bit 63 = H8).
    /// A set bit indicates a white piece on that square.
    #[must_use]
    fn white_bitboard(&self) -> u64 {
        self.inner.state.pieces[0]
    }

    /// Returns the black pieces bitboard as a u64.
    #[must_use]
    fn black_bitboard(&self) -> u64 {
        self.inner.state.pieces[1]
    }

    /// Returns the kings bitboard as a u64.
    ///
    /// This is a subset of white_bitboard | black_bitboard.
    #[must_use]
    fn kings_bitboard(&self) -> u64 {
        self.inner.state.kings
    }

    /// Returns all bitboards as a tuple: (white, black, kings, turn).
    ///
    /// Turn is 0 for White, 1 for Black.
    /// Useful for ML feature extraction.
    #[must_use]
    fn bitboards(&self) -> (u64, u64, u64, u8) {
        (
            self.inner.state.pieces[0],
            self.inner.state.pieces[1],
            self.inner.state.kings,
            self.inner.turn as u8,
        )
    }

    /// Returns the board state as a flat array of 4 u64 values.
    ///
    /// Order: [white_pieces, black_pieces, kings, turn_as_u64]
    /// Convenient for numpy: `np.array(board.to_array(), dtype=np.uint64)`
    #[must_use]
    fn to_array(&self) -> [u64; 4] {
        [
            self.inner.state.pieces[0],
            self.inner.state.pieces[1],
            self.inner.state.kings,
            self.inner.turn as u64,
        ]
    }

    /// Creates a board from raw bitboards.
    ///
    /// Args:
    ///     turn: The team to move (0 = White, 1 = Black).
    ///     white: White pieces bitboard.
    ///     black: Black pieces bitboard.
    ///     kings: Kings bitboard (must be subset of white | black).
    #[staticmethod]
    fn from_bitboards(turn: u8, white: u64, black: u64, kings: u64) -> PyResult<Self> {
        let team = match turn {
            0 => kish_core::Team::White,
            1 => kish_core::Team::Black,
            _ => return Err(PyValueError::new_err("turn must be 0 (White) or 1 (Black)")),
        };
        Ok(Self {
            inner: kish_core::Board::new(team, kish_core::State::new([white, black], kings)),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Board(turn={}, white={}, black={}, kings={})",
            self.turn().__repr__(),
            self.white_pieces().len(),
            self.black_pieces().len(),
            self.kings().len()
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl Board {
    fn mask_to_squares(mask: u64) -> Vec<Square> {
        let mut squares = Vec::new();
        let mut m = mask;
        while m != 0 {
            let idx = m.trailing_zeros() as u8;
            // SAFETY: trailing_zeros of non-zero u64 is 0-63
            squares.push(unsafe { kish_core::Square::from_u8(idx) }.into());
            m &= m - 1; // Clear lowest set bit
        }
        squares
    }
}

// ============================================================================
// Game
// ============================================================================

/// A mutable game with history tracking for proper draw detection.
///
/// `Game` wraps [`Board`] and adds history tracking for:
/// - **Threefold repetition**: Draw when same position occurs 3 times
/// - **150-move rule**: Draw after 150 moves without capture
/// - **Undo functionality**: Revert to previous positions
///
/// Use `Game` for playing full games. For pure move generation (AI search, perft),
/// use [`Board`] directly for better performance.
///
/// # Example
/// ```python
/// import kish
///
/// game = kish.Game()
///
/// # Play moves (mutates game state)
/// while not game.status().is_over():
///     actions = game.actions()
///     game.make_move(actions[0])
///
///     # Check draw conditions
///     if game.is_threefold_repetition():
///         print("Draw by repetition!")
///         break
///
/// # Undo moves
/// game.undo_move()
///
/// # Query state
/// print(f"Moves: {game.move_count}")
/// print(f"Halfmove clock: {game.halfmove_clock}")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct Game {
    inner: kish_core::Game,
}

#[pymethods]
impl Game {
    /// Creates a new game with the standard starting position.
    #[new]
    fn new() -> Self {
        Self {
            inner: kish_core::Game::new(),
        }
    }

    /// Creates a game from an existing board position.
    #[staticmethod]
    fn from_board(board: &Board) -> Self {
        Self {
            inner: kish_core::Game::from_board(board.inner),
        }
    }

    /// Returns the current board state.
    #[must_use]
    fn board(&self) -> Board {
        Board {
            inner: *self.inner.board(),
        }
    }

    /// Returns the current team's turn.
    #[getter]
    fn turn(&self) -> Team {
        self.inner.turn().into()
    }

    /// Returns all legal actions from the current position.
    #[must_use]
    fn actions(&self) -> Vec<Action> {
        let board = self.inner.board();
        let team = board.turn;
        self.inner
            .actions()
            .into_iter()
            .map(|action| {
                let detailed = action.to_detailed(team, &board.state);
                Action {
                    inner: action,
                    detailed,
                    team,
                }
            })
            .collect()
    }

    /// Returns the current game status (including draw conditions).
    #[must_use]
    fn status(&self) -> GameStatus {
        self.inner.status().into()
    }

    /// Makes a move and updates the game state.
    fn make_move(&mut self, action: &Action) {
        self.inner.make_move(&action.inner);
    }

    /// Undoes the last move. Returns true if a move was undone.
    fn undo_move(&mut self) -> bool {
        self.inner.undo_move()
    }

    /// Returns the number of half-moves since the last capture.
    #[getter]
    fn halfmove_clock(&self) -> u16 {
        self.inner.halfmove_clock()
    }

    /// Returns the number of moves made in this game.
    #[getter]
    fn move_count(&self) -> usize {
        self.inner.move_count()
    }

    /// Returns true if the current position has occurred 3+ times.
    #[must_use]
    fn is_threefold_repetition(&self) -> bool {
        self.inner.is_threefold_repetition()
    }

    /// Returns how many times the current position has occurred.
    #[must_use]
    fn position_count(&self) -> u8 {
        self.inner.position_occurrence_count()
    }

    /// Clears the game history and resets the halfmove clock.
    fn clear_history(&mut self) {
        self.inner.clear_history();
    }

    /// Runs a perft (performance test) at the given depth.
    #[must_use]
    fn perft(&mut self, depth: u64) -> u64 {
        self.inner.perft(depth)
    }

    fn __repr__(&self) -> String {
        format!(
            "Game(turn={}, moves={}, halfmove_clock={})",
            self.turn().__repr__(),
            self.move_count(),
            self.halfmove_clock()
        )
    }

    fn __str__(&self) -> String {
        self.board().__str__()
    }
}

// ============================================================================
// Module
// ============================================================================

/// Python bindings for the kish Turkish Draughts engine.
#[pymodule]
fn kish(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Team>()?;
    m.add_class::<Square>()?;
    m.add_class::<GameStatus>()?;
    m.add_class::<Action>()?;
    m.add_class::<Board>()?;
    m.add_class::<Game>()?;
    Ok(())
}
