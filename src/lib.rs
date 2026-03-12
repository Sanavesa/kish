//! # Kish - Turkish Checkers (Dama) Engine
//!
//! A high-performance Rust implementation of Turkish Draughts (Dama), featuring
//! bitboard-based state representation and optimized move generation.
//!
//! ## Overview
//!
//! Turkish Draughts, known as **Dama** (or **Türk Daması**) in Turkey, is a variant
//! of checkers distinguished by its **orthogonal movement** (horizontal and vertical)
//! rather than diagonal movement. This library implements all official rules including:
//!
//! - Orthogonal movement for men (forward, left, right) and kings (all four directions)
//! - Flying captures for kings (capture from any distance, land anywhere beyond)
//! - Mandatory capture rule (must capture if able)
//! - Maximum capture rule (must choose the sequence capturing the most pieces)
//! - 180-degree turn prohibition during multi-capture sequences
//! - Immediate piece removal during captures
//! - Promotion to king on the back row
//! - Mid-capture promotion rules (continues as pawn, promotes at turn end)
//!
//! ## Quick Start
//!
//! ```rust
//! use kish::{Board, Team, GameStatus};
//!
//! // Create a new game with the standard starting position
//! let board = Board::new_default();
//!
//! // White moves first
//! assert_eq!(board.turn, Team::White);
//!
//! // Get all legal actions
//! let actions = board.actions();
//! println!("White has {} legal moves", actions.len());
//!
//! // Apply an action to get the new board state
//! if let Some(action) = actions.first() {
//!     let mut new_board = board.apply(action);
//!     new_board.swap_turn_(); // Change to opponent's turn
//!
//!     // Check game status
//!     match new_board.status() {
//!         GameStatus::InProgress => println!("Game continues"),
//!         GameStatus::Draw => println!("Game is a draw"),
//!         GameStatus::Won(team) => println!("{} wins!", team),
//!     }
//! }
//! ```
//!
//! ## Board Representation
//!
//! The board uses a bitboard representation where each `u64` represents 64 squares:
//!
//! ```text
//!     a   b   c   d   e   f   g   h
//!   +---+---+---+---+---+---+---+---+
//! 8 |56 |57 |58 |59 |60 |61 |62 |63 |  ← Black's back row (White promotes here)
//!   +---+---+---+---+---+---+---+---+
//! 7 |48 |49 |50 |51 |52 |53 |54 |55 |  ← Black pieces start (rows 6-7)
//!   +---+---+---+---+---+---+---+---+
//! 6 |40 |41 |42 |43 |44 |45 |46 |47 |
//!   +---+---+---+---+---+---+---+---+
//! 5 |32 |33 |34 |35 |36 |37 |38 |39 |
//!   +---+---+---+---+---+---+---+---+
//! 4 |24 |25 |26 |27 |28 |29 |30 |31 |
//!   +---+---+---+---+---+---+---+---+
//! 3 |16 |17 |18 |19 |20 |21 |22 |23 |  ← White pieces start (rows 2-3)
//!   +---+---+---+---+---+---+---+---+
//! 2 | 8 | 9 |10 |11 |12 |13 |14 |15 |
//!   +---+---+---+---+---+---+---+---+
//! 1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  ← White's back row (Black promotes here)
//!   +---+---+---+---+---+---+---+---+
//!     a   b   c   d   e   f   g   h
//! ```
//!
//! ## Performance
//!
//! This engine is optimized for high-speed move generation and game tree search:
//!
//! - **Bitboard operations**: All piece positions stored as `u64` bitmasks
//! - **Compile-time generics**: Team-specific code paths via const generics
//! - **XOR-based deltas**: Actions store state changes, not full board copies
//! - **Lazy evaluation**: Capture detection uses early-exit optimizations
//!
//! ## Key Types
//!
//! - [`Board`]: The main game state containing piece positions and current turn
//! - [`Game`]: Full game with history tracking for draw detection (threefold repetition, 150-move rule)
//! - [`Action`]: A move represented as a bitboard delta (fast for simulations)
//! - [`ActionPath`]: A move with full path information (for UI/notation)
//! - [`Square`]: A single square on the board (0-63)
//! - [`Team`]: White or Black
//! - [`State`]: Raw bitboard state without turn information
//! - [`GameStatus`]: Current game status (`InProgress`, `Draw`, or `Won`)
//!
//! ## Move Notation
//!
//! The library supports standard algebraic notation for Turkish Draughts:
//!
//! | Move Type | Format | Example | Description |
//! |-----------|--------|---------|-------------|
//! | Non-capturing | `from-to` | `d3-d4` | Man moves forward |
//! | Single capture | `fromxto` | `d4xd6` | Piece jumps over enemy |
//! | Multi-capture | `fromxmidxto` | `d4xd6xf6` | Chain capture |
//! | Promotion | `from-to=K` | `d7-d8=K` | Man becomes king |
//!
//! ```rust
//! use kish::{Board, Team, Square};
//!
//! let board = Board::from_squares(
//!     Team::White,
//!     &[Square::D4],
//!     &[Square::D5],
//!     &[],
//! );
//!
//! let actions = board.actions();
//! let detailed = actions[0].to_detailed(board.turn, &board.state);
//! assert_eq!(detailed.to_notation(), "d4xd6"); // Capture notation
//! ```
//!
//! ## Rules Reference
//!
//! For complete rules documentation, see the `docs/rules.md` file in the repository.
//!
//! ### Key Rules Summary
//!
//! 1. **Movement**: Men move 1 square forward/left/right. Kings move any distance orthogonally.
//! 2. **Captures are mandatory**: If you can capture, you must.
//! 3. **Maximum capture**: Must choose the path capturing the most pieces.
//! 4. **No 180° turns**: During multi-capture, can't reverse direction.
//! 5. **One piece each = Draw**: When both players have exactly one piece.
//! 6. **Blocking = Loss**: If you can't move, you lose.

mod action;
mod actiongen;
mod board;
mod game;
mod game_status;
mod perft;
mod square;
mod state;
mod team;

pub use action::{Action, ActionPath};
pub use board::Board;
pub use game::Game;
pub use game_status::GameStatus;
pub use square::Square;
pub use state::State;
pub use team::Team;
