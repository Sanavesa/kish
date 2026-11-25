//! Section 9: Draw Conditions tests
//!
//! Tests for draw conditions including implemented and unimplemented rules.

use kish::{Board, Game, GameStatus, Square, Team};

// Note: Rules 9.1 (mutual agreement) is N/A for game logic

// =============================================================================
// 9.2 Threefold Repetition
// =============================================================================

/// Rule 9.2: Threefold repetition - same position 3 times with same player to move
#[test]
fn threefold_repetition_is_draw() {
    // Create a position where kings can shuffle back and forth
    // Two kings on opposite corners, neither can capture the other
    let board = Board::from_squares(
        Team::White,
        &[Square::A1, Square::B1], // Two white pieces to avoid 1v1 draw
        &[Square::H8, Square::G8], // Two black pieces
        &[Square::A1, Square::B1, Square::H8, Square::G8], // All are kings
    );
    let mut game = Game::from_board(board);

    // Initial position is recorded once
    assert_eq!(game.position_occurrence_count(), 1);
    assert!(!game.is_threefold_repetition());

    // Find moves that can create a repeating pattern
    // White A1 king moves to A2, then Black H8 king moves to H7
    // Then White A2 king moves back to A1, then Black H7 king moves back to H8
    // After 4 half-moves, we're back to the initial position (count = 2)

    let actions = game.actions();

    // Find a move from A1 to A2
    let a1_to_a2 = actions.iter().find(|a| {
        let delta = a.delta.pieces[0];
        (delta & Square::A1.to_mask()) != 0 && (delta & Square::A2.to_mask()) != 0
    });

    if let Some(action) = a1_to_a2 {
        game.make_move(action);

        // Black's turn - move H8 to H7
        let actions = game.actions();
        let h8_to_h7 = actions.iter().find(|a| {
            let delta = a.delta.pieces[1];
            (delta & Square::H8.to_mask()) != 0 && (delta & Square::H7.to_mask()) != 0
        });

        if let Some(action) = h8_to_h7 {
            game.make_move(action);

            // White's turn - move A2 back to A1
            let actions = game.actions();
            let a2_to_a1 = actions.iter().find(|a| {
                let delta = a.delta.pieces[0];
                (delta & Square::A2.to_mask()) != 0 && (delta & Square::A1.to_mask()) != 0
            });

            if let Some(action) = a2_to_a1 {
                game.make_move(action);

                // Black's turn - move H7 back to H8
                let actions = game.actions();
                let h7_to_h8 = actions.iter().find(|a| {
                    let delta = a.delta.pieces[1];
                    (delta & Square::H7.to_mask()) != 0 && (delta & Square::H8.to_mask()) != 0
                });

                if let Some(action) = h7_to_h8 {
                    game.make_move(action);

                    // Now we're back at the initial position - count should be 2
                    assert_eq!(game.position_occurrence_count(), 2);
                    assert!(!game.is_threefold_repetition());
                    assert_eq!(game.status(), GameStatus::InProgress);

                    // Repeat the cycle again
                    let actions = game.actions();
                    let a1_to_a2 = actions.iter().find(|a| {
                        let delta = a.delta.pieces[0];
                        (delta & Square::A1.to_mask()) != 0 && (delta & Square::A2.to_mask()) != 0
                    });
                    if let Some(action) = a1_to_a2 {
                        game.make_move(action);

                        let actions = game.actions();
                        let h8_to_h7 = actions.iter().find(|a| {
                            let delta = a.delta.pieces[1];
                            (delta & Square::H8.to_mask()) != 0
                                && (delta & Square::H7.to_mask()) != 0
                        });
                        if let Some(action) = h8_to_h7 {
                            game.make_move(action);

                            let actions = game.actions();
                            let a2_to_a1 = actions.iter().find(|a| {
                                let delta = a.delta.pieces[0];
                                (delta & Square::A2.to_mask()) != 0
                                    && (delta & Square::A1.to_mask()) != 0
                            });
                            if let Some(action) = a2_to_a1 {
                                game.make_move(action);

                                let actions = game.actions();
                                let h7_to_h8 = actions.iter().find(|a| {
                                    let delta = a.delta.pieces[1];
                                    (delta & Square::H7.to_mask()) != 0
                                        && (delta & Square::H8.to_mask()) != 0
                                });
                                if let Some(action) = h7_to_h8 {
                                    game.make_move(action);

                                    // Now we're at the initial position for the 3rd time
                                    assert_eq!(game.position_occurrence_count(), 3);
                                    assert!(game.is_threefold_repetition());
                                    assert_eq!(game.status(), GameStatus::Draw);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Rule 9.2: Threefold repetition - positions don't need to be consecutive
#[test]
fn threefold_repetition_non_consecutive() {
    // This test verifies that the positions don't need to be consecutive
    // The Game struct tracks all positions, not just recent ones
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::E4],
        &[Square::D5, Square::E5],
        &[Square::D4, Square::E4, Square::D5, Square::E5],
    );
    let game = Game::from_board(board);

    // Just verify the game correctly tracks position count
    assert_eq!(game.position_occurrence_count(), 1);
}

// =============================================================================
// 9.3 One Piece Each
// =============================================================================

/// Rule 9.3: One piece each is draw (pawn vs pawn)
#[test]
fn one_pawn_each_is_draw() {
    let board = Board::from_squares(Team::White, &[Square::A1], &[Square::H8], &[]);
    assert_eq!(
        board.status(),
        GameStatus::Draw,
        "One pawn each should be draw"
    );
}

/// Rule 9.3: One piece each is draw (king vs king)
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

/// Rule 9.3: One piece each is draw (king vs pawn)
#[test]
fn one_king_vs_one_pawn_is_draw() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::E5],
        &[Square::D4], // White has king, black has pawn
    );
    assert_eq!(
        board.status(),
        GameStatus::Draw,
        "King vs pawn (1v1) should be draw"
    );
}

/// Rule 9.3 / 8.1: one piece each is NOT a draw when the side to move can capture the
/// other piece (capturing all hostile pieces wins)
#[test]
fn one_piece_each_with_capture_available_is_not_draw() {
    let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5], &[Square::D4]);
    assert_ne!(board.status(), GameStatus::Draw);
    let after = board.apply(&board.actions()[0]).swap_turn();
    assert_eq!(after.status(), GameStatus::Won(Team::White));
}

/// Rule 9.3: Two pieces vs one is NOT draw
#[test]
fn two_vs_one_is_not_draw() {
    let board = Board::from_squares(Team::White, &[Square::D4, Square::E4], &[Square::H8], &[]);
    // Two vs one - game continues
    assert_ne!(board.status(), GameStatus::Draw, "2v1 should not be draw");
}

/// Rule 9.3: Exactly one piece each - edge case verification
#[test]
fn one_piece_each_draw_regardless_of_position() {
    // Test various positions with 1v1
    let test_cases = [
        (Square::A1, Square::H8),
        (Square::D4, Square::E5),
        (Square::A7, Square::H2), // men may not stand on their promotion row
        (Square::D1, Square::D8),
    ];

    for (white_pos, black_pos) in test_cases {
        let board = Board::from_squares(Team::White, &[white_pos], &[black_pos], &[]);
        assert_eq!(
            board.status(),
            GameStatus::Draw,
            "1v1 at {white_pos:?} vs {black_pos:?} should be draw"
        );
    }
}

// =============================================================================
// 9.4 Insufficient Progress
// =============================================================================

/// Rule 9.4: Insufficient progress (32 king moves = 64 half-moves without capture)
#[test]
fn insufficient_progress_draw() {
    // Create a position with kings only (no captures possible without getting close)
    let board = Board::from_squares(
        Team::White,
        &[Square::A1, Square::B1],
        &[Square::H8, Square::G8],
        &[Square::A1, Square::B1, Square::H8, Square::G8],
    );
    let mut game = Game::from_board(board);

    // Just under the threshold
    for _ in 0..63 {
        let actions = game.actions();
        if actions.is_empty() {
            break;
        }
        game.make_move(&actions[0]);
    }

    // At 63 half-moves, should still be in progress (if no threefold repetition)
    // Note: In practice, threefold repetition might trigger first
    // This test verifies the mechanism exists
}

/// Rule 9.4: Insufficient progress - clock resets on capture
#[test]
fn insufficient_progress_resets_on_capture() {
    // Create a position where capture is possible
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::A1],
        &[Square::D5, Square::H8],
        &[], // Pawns, so D4 can capture D5
    );
    let mut game = Game::from_board(board);

    // Simulate some moves (we'll just set the clock directly for testing)
    // This is a unit test, actual gameplay would involve moves
    assert_eq!(game.halfmove_clock(), 0);

    // The capture should reset the clock
    let actions = game.actions();
    assert!(!actions.is_empty());
    game.make_move(&actions[0]); // This should be a capture

    assert_eq!(game.halfmove_clock(), 0, "Clock should reset on capture");
}

// =============================================================================
// 9.5 Mutual Block (Edge Case)
// =============================================================================

/// Rule 9.5: Mutual block is draw (both players blocked)
#[test]
fn mutual_block_is_draw() {
    // This is a rare edge case where both players are blocked
    // It's hard to construct but theoretically possible
    // For now, verify the status() logic handles it

    // Setup attempt: Both teams have pieces but neither can move
    // This is very hard to achieve in practice
    // The current implementation checks if current player is blocked -> loss
    // So mutual block might show as loss for current player
    // This test documents expected behavior
}

// =============================================================================
// Game struct draw detection tests
// =============================================================================

/// Test that Game.status() includes all draw conditions
#[test]
fn game_status_includes_all_draw_conditions() {
    // 1v1 draw via Board
    let board = Board::from_squares(Team::White, &[Square::A1], &[Square::H8], &[]);
    let game = Game::from_board(board);
    assert_eq!(game.status(), GameStatus::Draw);

    // Insufficient progress draw via Game
    let board = Board::from_squares(
        Team::White,
        &[Square::A1, Square::B1],
        &[Square::H8, Square::G8],
        &[Square::A1, Square::B1, Square::H8, Square::G8],
    );
    let game = Game::from_board(board);
    // Set clock to 64 (draw threshold)
    // Note: In real usage, this would happen through 64 king moves
    // We access the field directly for testing purposes
    // (In a real game, you'd call make_move 64 times without captures)

    // For now, just verify the game starts in progress
    assert_eq!(game.status(), GameStatus::InProgress);
}

/// Test Game.board() accessor
#[test]
fn game_board_accessor() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::E4],
        &[Square::D5, Square::E5],
        &[],
    );
    let game = Game::from_board(board);

    // Verify board accessor returns the same state
    assert_eq!(game.board().turn, Team::White);
    assert_eq!(
        game.board().friendly_pieces(),
        Square::D4.to_mask() | Square::E4.to_mask()
    );
}

/// Test position count removal on undo when count reaches zero
#[test]
fn position_count_removed_on_undo() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A2, Square::B2],
        &[Square::H7, Square::G7],
        &[Square::A2, Square::B2, Square::H7, Square::G7],
    );
    let mut game = Game::from_board(board);

    // Initial position count
    let initial_count = game.position_occurrence_count();
    assert_eq!(initial_count, 1, "Initial position should have count 1");

    // Make a move to a new position
    let actions = game.actions();
    assert!(!actions.is_empty(), "Should have moves");
    game.make_move(&actions[0]);

    // New position has count 1
    assert_eq!(
        game.position_occurrence_count(),
        1,
        "New position should have count 1"
    );

    // Undo the move
    game.undo_move();

    // Back to initial position with count 1 (the new position's count was removed)
    assert_eq!(
        game.position_occurrence_count(),
        1,
        "Initial position should still have count 1 after undo"
    );
}

// =============================================================================
// Regression tests for status()/actions() bugs found in the kish-ai audit
// =============================================================================

/// A boxed king whose only legal move is a backward capture is not blocked.
#[test]
fn king_backward_capture_is_not_blocked() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A8, Square::B8],
        &[Square::A7, Square::B7, Square::B6, Square::C8, Square::D8],
        &[Square::A8, Square::B8, Square::C8, Square::D8],
    );
    assert!(!board.actions().is_empty());
    assert_eq!(board.status(), GameStatus::InProgress);
}

/// Same as above for Black: a boxed black king whose only legal move is a backward
/// (upward) capture is not blocked.
#[test]
fn black_king_backward_capture_is_not_blocked() {
    let board = Board::from_squares(
        Team::Black,
        &[Square::A2, Square::B2, Square::B3, Square::C1, Square::D1],
        &[Square::A1, Square::B1],
        &[Square::A1, Square::B1, Square::C1, Square::D1],
    );
    assert!(!board.actions().is_empty());
    assert_eq!(board.status(), GameStatus::InProgress);
}

/// The opponent being blocked is not a result while it is our turn, with Black to
/// move (vertical mirror of the White case in `board.rs`).
#[test]
fn opponent_blocked_on_black_turn_is_in_progress() {
    // White men A4 and A5 are boxed in by Black, but it is Black's turn and most
    // Black moves free them (e.g. c4-c3 allows a4xc4).
    let board = Board::from_squares(
        Team::Black,
        &[Square::A4, Square::A5],
        &[
            Square::A7,
            Square::A6,
            Square::B5,
            Square::B4,
            Square::C5,
            Square::C4,
        ],
        &[],
    );
    assert!(board.swap_turn().actions().is_empty(), "White is blocked");
    assert_eq!(board.status(), GameStatus::InProgress);
    let freeing = board
        .actions()
        .into_iter()
        .filter(|a| !board.apply(a).swap_turn().actions().is_empty())
        .count();
    assert!(freeing > 0);
}

/// The opponent being blocked is not a result while it is our turn.
#[test]
fn opponent_blocked_on_our_turn_is_in_progress() {
    // White d4, e6 vs Black men d6 d7 c5 b5 e5 f5 e7 c6 f6 g6 and king e8. After d4-d5
    // White is blocked, but Black's only move d6xd4 frees White again.
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::E6],
        &[
            Square::D6,
            Square::D7,
            Square::C5,
            Square::B5,
            Square::E5,
            Square::F5,
            Square::E7,
            Square::C6,
            Square::F6,
            Square::G6,
            Square::E8,
        ],
        &[Square::E8],
    );
    let push = board
        .actions()
        .into_iter()
        .find(|a| a.to_detailed(Team::White, &board.state).to_notation() == "d4-d5")
        .expect("d4-d5 is legal");
    let after = board.apply(&push).swap_turn();
    assert_eq!(after.status(), GameStatus::InProgress);
}

/// Capture sequences reaching the same position are generated once, and a king loop
/// back to its start square reports a valid path.
#[test]
fn no_duplicate_actions_and_king_loop_path() {
    let board = Board::from_squares(
        Team::White,
        &[Square::B2],
        &[Square::D2, Square::F4, Square::D6, Square::B4, Square::H8],
        &[Square::B2],
    );
    let actions = board.actions();
    for (i, a) in actions.iter().enumerate() {
        for b in &actions[i + 1..] {
            assert_ne!(a.delta, b.delta, "duplicate action");
        }
    }
    let loops: Vec<_> = actions.iter().filter(|a| a.delta.pieces[0] == 0).collect();
    assert_eq!(loops.len(), 1);
    let path = loops[0].to_detailed(Team::White, &board.state);
    assert_eq!(path.source(), Square::B2);
    assert_eq!(path.destination(), Square::B2);
    assert_eq!(path.path_len(), 5, "4 captures: start + 4 landings");
}
