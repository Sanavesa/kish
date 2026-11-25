//! Section 8: Winning the Game tests
//!
//! Tests for win conditions (elimination and blocking).

use kish::{Board, GameStatus, Square, Team};

/// Rule 8.1: Capture all opponent's pieces - white wins
#[test]
fn white_wins_by_capturing_all_black_pieces() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::E4],
        &[], // Black has no pieces
        &[],
    );
    assert_eq!(
        board.status(),
        GameStatus::Won(Team::White),
        "White wins when black has no pieces"
    );
}

/// Rule 8.1: Capture all opponent's pieces - black wins
#[test]
fn black_wins_by_capturing_all_white_pieces() {
    let board = Board::from_squares(
        Team::White,
        &[], // White has no pieces
        &[Square::D4, Square::E4],
        &[],
    );
    assert_eq!(
        board.status(),
        GameStatus::Won(Team::Black),
        "Black wins when white has no pieces"
    );
}

/// Rule 8.2: Block all opponent's pieces - opponent loses
#[test]
fn blocked_player_loses_white_blocked() {
    // White pawns completely blocked by black pieces
    let board = Board::from_squares(
        Team::White,
        &[Square::A2, Square::A3],
        &[
            Square::A4, // Blocks A3's forward
            Square::A5, // Extra blocker
            Square::B2, // Adjacent to A2
            Square::B3, // Adjacent to A3
            Square::C2, // Blocks B2 capture landing
            Square::C3, // Blocks B3 capture landing
        ],
        &[],
    );
    assert_eq!(
        board.status(),
        GameStatus::Won(Team::Black),
        "Blocked white should lose"
    );
}

/// Rule 8.2: Block all opponent's pieces - black blocked
#[test]
fn blocked_player_loses_black_blocked() {
    // Black pieces blocked, Black to move (being blocked only loses on your own turn)
    let board = Board::from_squares(
        Team::Black,
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
    assert_eq!(
        board.status(),
        GameStatus::Won(Team::White),
        "Blocked black should lose"
    );
}

/// Rule 8: King can't move means loss (even if pawns exist)
#[test]
fn single_blocked_king_loses() {
    // White king at A1, completely surrounded by black pieces
    // with no escape and no capture possible
    let board = Board::from_squares(
        Team::White,
        &[Square::A1],
        &[
            Square::A2,
            Square::B1,
            Square::B2,
            Square::C1,
            Square::A3,
            Square::C2,
        ],
        &[Square::A1],
    );
    // A1 king: right blocked by B1, up blocked by A2
    // Capture B1? Landing at C1 blocked
    // Capture A2? Landing at A3 blocked
    assert_eq!(
        board.status(),
        GameStatus::Won(Team::Black),
        "Completely blocked king should lose"
    );
}

/// Rule 8: Game in progress when both can move
#[test]
fn game_in_progress_when_both_can_move() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::E4],
        &[Square::D5, Square::E5],
        &[],
    );
    // White must capture D5 or E5, black can respond
    assert_eq!(
        board.status(),
        GameStatus::InProgress,
        "Game should be in progress"
    );
}

// =============================================================================
// 8.2 Blocking - Edge cases for is_blocked coverage
// =============================================================================

/// Rule 8.2: Black king only has backward (up) moves available
#[test]
fn black_king_not_blocked_with_only_backward_moves() {
    // Black king at H1 (corner), only can move backward (up)
    // Left: G1 blocked by white piece
    // Right: edge (column H)
    // Forward (down for black): edge (row 1)
    // Backward (up for black): H2, H3, etc. available
    let board = Board::from_squares(
        Team::Black,
        &[Square::G1, Square::A8], // G1 blocks left, A8 avoids 1v1 draw
        &[Square::H1],
        &[Square::H1], // H1 is a black king
    );
    // Should be in progress - black king can only move backward (up)
    assert_eq!(
        board.status(),
        GameStatus::InProgress,
        "Black king with only backward moves should not be blocked"
    );
}

/// Rule 8.2: Black pawn only has vertical (down) capture available
#[test]
fn black_pawn_not_blocked_with_only_vertical_capture() {
    // Need to create a scenario where:
    // 1. No moves possible (left/right/forward blocked for ALL pieces)
    // 2. No left/right captures possible
    // 3. Only vertical capture possible
    //
    // Black pawn at A3:
    // - Left move: edge (column A)
    // - Right move: B3 blocked by white piece
    // - Forward (down): A2 blocked by white piece (also capturable)
    // - Left capture: edge
    // - Right capture: B3 has white, but C3 blocked (no landing)
    // - Vertical capture: A2 is hostile, A1 is empty (landing available)
    let board = Board::from_squares(
        Team::Black,
        &[Square::A2, Square::B3, Square::C3], // Block all movements, A2 capturable
        &[Square::A3],                         // Single black pawn
        &[],
    );
    // A3 pawn:
    // - left=edge
    // - right=B3 white (blocked)
    // - forward=A2 white (blocked/capturable)
    // - backward=A4 empty but pawns can't move backward
    // - left capture: edge
    // - right capture: B3 white, but C3 blocked (can't land)
    // - vertical capture: A2 white, A1 empty (can capture!)
    assert_eq!(
        board.status(),
        GameStatus::InProgress,
        "Black pawn with only vertical capture should not be blocked"
    );
}
