//! Game status representation for Turkish Draughts.
//!
//! This module defines the [`GameStatus`] enum representing the current state of a game:
//! - **InProgress**: The game is ongoing
//! - **Draw**: The game ended in a draw (e.g., 1v1 piece scenario)
//! - **Won(Team)**: A player has won
//!
//! # Win Conditions
//!
//! A player wins when their opponent:
//! - Has no pieces remaining
//! - Has no legal moves (is blocked)
//!
//! # Draw Conditions
//!
//! The game is a draw when:
//! - Both players have exactly one piece each (Rule 9.3)
//! - Threefold repetition occurs (Rule 9.2) - tracked by [`Game`](crate::Game)
//! - 150 consecutive plies without capture (Rule 9.4) - tracked by [`Game`](crate::Game)
//!
//! # Example
//!
//! ```rust
//! use kish::{Board, GameStatus, Team};
//!
//! let board = Board::new_default();
//! assert_eq!(board.status(), GameStatus::InProgress);
//! assert!(!board.status().is_over());
//! ```

use std::fmt;

use super::Team;

/// Represents the current status of a game.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GameStatus {
    /// The game is in progress.
    InProgress,
    /// The game ended in a draw.
    Draw,
    /// The game was won by a team.
    Won(Team),
}

impl GameStatus {
    /// Checks if the game is over.
    #[inline]
    #[must_use]
    pub const fn is_over(&self) -> bool {
        !matches!(self, Self::InProgress)
    }
}

impl Default for GameStatus {
    /// Returns [`GameStatus::InProgress`] as the default, since all games start in progress.
    #[inline]
    fn default() -> Self {
        Self::InProgress
    }
}

impl fmt::Display for GameStatus {
    /// Formats the game status as a string.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InProgress => write!(f, "In Progress"),
            Self::Draw => write!(f, "Draw"),
            Self::Won(team) => write!(f, "Won by {team}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_over_returns_false_for_in_progress() {
        assert!(!GameStatus::InProgress.is_over());
    }

    #[test]
    fn is_over_returns_true_for_draw() {
        assert!(GameStatus::Draw.is_over());
    }

    #[test]
    fn is_over_returns_true_for_won() {
        assert!(GameStatus::Won(Team::White).is_over());
        assert!(GameStatus::Won(Team::Black).is_over());
    }

    #[test]
    fn display_in_progress() {
        assert_eq!(format!("{}", GameStatus::InProgress), "In Progress");
    }

    #[test]
    fn display_draw() {
        assert_eq!(format!("{}", GameStatus::Draw), "Draw");
    }

    #[test]
    fn display_won() {
        assert_eq!(format!("{}", GameStatus::Won(Team::White)), "Won by White");
        assert_eq!(format!("{}", GameStatus::Won(Team::Black)), "Won by Black");
    }

    #[test]
    fn default_is_in_progress() {
        assert_eq!(GameStatus::default(), GameStatus::InProgress);
    }
}
