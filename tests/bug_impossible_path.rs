//! Test for bug: impossible capture path generation
//!
//! ## Bug Description
//! The library generated geometrically impossible capture sequences.
//! Example: C8XC5XA5XH1XF1XH1XH5XH1
//!
//! Problems observed:
//! 1. A5→H1 is geometrically impossible (not on same row or column)
//! 2. H1 is visited 3 times in the path
//!
//! ## Root Cause
//! The `Action` struct stores moves as XOR deltas for efficiency, containing only:
//! - Source square (derived from delta)
//! - Final destination square (derived from delta)
//! - Captured pieces bitboard
//!
//! Critically, **intermediate landing squares are NOT stored**. When `to_detailed()`
//! reconstructed the path, it used a greedy algorithm that picked the first valid
//! landing square after each capture. For king multi-captures, this failed because:
//!
//! 1. Kings can land on ANY empty square beyond a captured piece (not just one)
//! 2. The greedy choice didn't verify remaining captures could be completed
//! 3. Choosing the wrong intermediate landing made subsequent captures unreachable
//!
//! ## Fix
//! Replaced the greedy algorithm with backtracking in `reconstruct_capture_path()`:
//! - Try all possible landing squares after each capture
//! - Recursively verify remaining captures can be completed
//! - Backtrack if a path doesn't lead to the final destination

use kish::{Board, Game, Square, State, Team};

/// Validates that a move path is geometrically valid for Turkish Checkers.
/// All moves must be horizontal or vertical (no diagonal), and a square may only be
/// revisited by a king (e.g. a capture loop back to its start square). Whether each
/// step is a legal capture on the board is checked by `replay_capture_path`.
fn validate_path(path: &[Square]) -> Result<(), String> {
    for i in 1..path.len() {
        let prev = &path[i - 1];
        let square = &path[i];
        if prev == square {
            return Err(format!("Path stays on {} between two landings", square));
        }

        // Turkish Checkers: moves must be horizontal or vertical
        if prev.row() != square.row() && prev.column() != square.column() {
            return Err(format!(
                "Invalid move {}→{}: not on same row or column (diagonal moves not allowed)",
                prev, square
            ));
        }
    }

    Ok(())
}

#[test]
fn test_impossible_capture_path_bug() {
    // Board state where the bug occurs
    let white: u64 = 288230376571242752;
    let black: u64 = 23104780763204;
    let kings: u64 = 288230376151711812;

    /*
    Board visualization:
        A B C D E F G H
      +-----------------+
    8 | . . W . . . . . | 8   (W = White King at C8)
    7 | . . . . . . . . | 7
    6 | b . b . b . . . | 6   (b = black pawn)
    5 | b b . . . . . . | 5
    4 | w . . w w . . b | 4   (w = white pawn)
    3 | w . . . . . b . | 3
    2 | w . . w . . . w | 2
    1 | . . B . . . B . | 1   (B = Black King)
      +-----------------+
        A B C D E F G H
    */

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);
    let actions = game.actions();

    println!("Legal moves ({} total):", actions.len());

    let mut found_bug = false;

    for action in &actions {
        let detailed = action.to_detailed(Team::White, &board.state);
        let path = detailed.path();
        let notation = detailed.to_notation();

        let path_strs: Vec<_> = path.iter().map(|s| s.to_string()).collect();
        println!("  Move: {} (path: {:?})", notation, path_strs);

        if let Err(e) = validate_path(path) {
            println!("    *** BUG: {} ***", e);
            found_bug = true;
        }
    }

    assert!(
        !found_bug,
        "Found geometrically impossible moves - see output above"
    );
}

#[test]
fn test_a5_to_h1_impossible() {
    // Direct test: verify A5→H1 is NOT on same row or column
    let a5 = Square::A5;
    let h1 = Square::H1;

    let a5_row = a5.row(); // 4 (0-indexed)
    let a5_col = a5.column(); // 0
    let h1_row = h1.row(); // 0
    let h1_col = h1.column(); // 7

    // A5 and H1 are NOT on same row and NOT on same column
    assert_ne!(a5_row, h1_row, "A5 and H1 should NOT be on same row");
    assert_ne!(a5_col, h1_col, "A5 and H1 should NOT be on same column");

    // Therefore, a direct move A5→H1 is impossible in Turkish Checkers
    println!("A5 is at row={}, col={}", a5_row, a5_col);
    println!("H1 is at row={}, col={}", h1_row, h1_col);
    println!(
        "A5→H1 requires moving {} rows and {} columns",
        (h1_row as i8 - a5_row as i8).abs(),
        (h1_col as i8 - a5_col as i8).abs()
    );
}

// =============================================================================
// Comprehensive path reconstruction tests
// =============================================================================

/// Test that ALL actions from ANY position can be converted to valid detailed paths.
/// This is a comprehensive "fuzz-like" test that validates the path reconstruction
/// algorithm works for all generated actions.
#[test]
fn all_actions_have_valid_paths() {
    // Test the default starting position
    let game = Game::default();
    validate_all_actions_have_valid_paths(&game);
}

/// Test path reconstruction with a complex mid-game position.
#[test]
fn midgame_position_all_actions_valid() {
    // A complex mid-game position with kings and pawns
    let white = Square::A2.to_mask() | Square::D4.to_mask() | Square::H8.to_mask();
    let black =
        Square::B3.to_mask() | Square::C4.to_mask() | Square::E5.to_mask() | Square::F6.to_mask();
    let kings = Square::H8.to_mask(); // White king at H8

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);
    validate_all_actions_have_valid_paths(&game);
}

/// Test king with multiple capture opportunities requiring specific landing choices.
/// This tests the scenario where the greedy algorithm would fail.
#[test]
fn king_multi_capture_requires_specific_landing() {
    // White king at A1, black pieces arranged so that only specific landing
    // squares lead to a complete capture sequence
    //
    //     A B C D E F G H
    //   +-----------------+
    // 8 | . . . . . . . . |
    // 7 | . . . . . . . . |
    // 6 | . . . . . . . . |
    // 5 | . . . b . . . . |  b = black pawn at D5
    // 4 | . . . . . . . . |
    // 3 | . b . . . . . . |  b = black pawn at B3
    // 2 | . . . . . . . . |
    // 1 | W . . . . . . . |  W = white king at A1
    //   +-----------------+
    //
    // King at A1 can capture B3, but must land at C3 (not D3, E3, etc.)
    // to then be able to capture D5 and continue.

    let white = Square::A1.to_mask();
    let black = Square::B3.to_mask() | Square::D5.to_mask();
    let kings = Square::A1.to_mask();

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Test king captures along board edges where landing options are limited.
#[test]
fn king_edge_captures() {
    // White king on edge, capturing along the edge
    //
    //     A B C D E F G H
    //   +-----------------+
    // 8 | W . b . b . b . |  W=king, b=black pawns
    // 7 | . . . . . . . . |
    // ...

    let white = Square::A8.to_mask();
    let black = Square::C8.to_mask() | Square::E8.to_mask() | Square::G8.to_mask();
    let kings = Square::A8.to_mask();

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Test king captures in corner positions.
#[test]
fn king_corner_captures() {
    // King in corner with captures going both directions
    let white = Square::A1.to_mask();
    let black = Square::A3.to_mask() | Square::C1.to_mask();
    let kings = Square::A1.to_mask();

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Test a long king capture chain (5+ captures).
#[test]
fn king_long_capture_chain() {
    // Arrange pieces for a long capture chain
    //
    //     A B C D E F G H
    //   +-----------------+
    // 8 | W . . . . . . . |  W = white king
    // 7 | . . . . . . . . |
    // 6 | b . . . . . b . |  b = black pawns
    // 5 | . . . . . . . . |
    // 4 | b . . . . . b . |
    // 3 | . . . . . . . . |
    // 2 | b . . . . . . . |
    // 1 | . . . . . . . . |
    //   +-----------------+

    let white = Square::A8.to_mask();
    let black = Square::A6.to_mask()
        | Square::A4.to_mask()
        | Square::A2.to_mask()
        | Square::G6.to_mask()
        | Square::G4.to_mask();
    let kings = Square::A8.to_mask();

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Test black king multi-captures (ensure team handling is correct).
#[test]
fn black_king_multi_capture() {
    let white = Square::B3.to_mask() | Square::D5.to_mask();
    let black = Square::A1.to_mask();
    let kings = Square::A1.to_mask();

    let state = State::new([white, black], kings);
    let board = Board::new(Team::Black, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Test pawn multi-captures (pawns have fixed landing squares, simpler case).
#[test]
fn pawn_multi_capture_path() {
    // White pawn at A3, black pieces at B3 and D3
    // Pawn captures: A3 -> C3 (over B3) -> E3 (over D3)
    let white = Square::A3.to_mask();
    let black = Square::B3.to_mask() | Square::D3.to_mask();
    let kings = 0;

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Test mixed scenario: some pawns, some kings.
#[test]
fn mixed_pieces_captures() {
    let white = Square::A1.to_mask() | Square::A3.to_mask(); // King at A1, pawn at A3
    let black = Square::B3.to_mask() | Square::C3.to_mask() | Square::A5.to_mask();
    let kings = Square::A1.to_mask();

    let state = State::new([white, black], kings);
    let board = Board::new(Team::White, state);
    let game = Game::from_board(board);

    validate_all_actions_have_valid_paths(&game);
}

/// Stress test: Generate many random-ish positions and validate all paths.
#[test]
fn stress_test_path_reconstruction() {
    // Test multiple positions derived from playing out games
    let mut game = Game::default();

    // Play some moves and validate at each step
    for _ in 0..20 {
        validate_all_actions_have_valid_paths(&game);

        let actions = game.actions();
        if actions.is_empty() {
            break;
        }

        // Pick first action and apply
        game.make_move(&actions[0]);
    }
}

/// Test the specific pattern that caused the original bug:
/// King needs to make a specific landing choice to reach all captures.
#[test]
fn original_bug_pattern_variations() {
    // Variation 1: Similar to original but with different piece arrangement
    let test_cases = [
        // (white_pieces, black_pieces, kings, turn)
        (
            Square::C8.to_mask(),
            Square::C6.to_mask()
                | Square::C4.to_mask()
                | Square::A4.to_mask()
                | Square::A2.to_mask(),
            Square::C8.to_mask(),
            Team::White,
        ),
        (
            Square::H1.to_mask(),
            Square::H3.to_mask() | Square::F3.to_mask() | Square::D3.to_mask(),
            Square::H1.to_mask(),
            Team::White,
        ),
        (
            Square::A1.to_mask(),
            Square::A3.to_mask()
                | Square::A5.to_mask()
                | Square::C5.to_mask()
                | Square::E5.to_mask(),
            Square::A1.to_mask(),
            Team::White,
        ),
    ];

    for (white, black, kings, turn) in test_cases {
        let state = State::new([white, black], kings);
        let board = Board::new(turn, state);
        let game = Game::from_board(board);
        validate_all_actions_have_valid_paths(&game);
    }
}

// =============================================================================
// Paths must respect board occupancy (fixed in v1.1.8)
// =============================================================================
//
// `to_detailed()` used to rebuild paths from the captured pieces alone, so a king's
// path could pass through or land on pieces that stay on the board. The move itself
// (its delta) was correct; only the reported path/notation was wrong.

/// A king's path must not land on a piece that is not captured.
#[test]
fn king_path_does_not_land_on_uncaptured_piece() {
    //     A B C D E F G H
    //   +-----------------+
    // 8 | . . . . . . . . |
    // 7 | . . . . . W . . |  W = white king at F7
    // 6 | . . . . . . . . |
    // 5 | . . . . . b b . |
    // 4 | . . . b . . . . |
    // 3 | b . . . . . . . |
    // 2 | . b . . . . b . |  G2 is not captured
    // 1 | . . . . . . . . |
    //   +-----------------+
    //
    // Before the fix the reported path was f7xf4xa4xa2xg2xg6: it lands on G2.
    let board = Board::from_squares(
        Team::White,
        &[Square::F7],
        &[
            Square::B2,
            Square::G2,
            Square::A3,
            Square::D4,
            Square::F5,
            Square::G5,
        ],
        &[Square::F7],
    );
    let notations: Vec<_> = board
        .actions()
        .iter()
        .map(|a| a.to_detailed(Team::White, &board.state).to_notation())
        .collect();
    assert!(
        notations.contains(&"f7xf2xa2xa4xg4xg6".to_string()),
        "{notations:?}"
    );
    assert!(
        !notations.iter().any(|n| n.contains("g2")),
        "a path lands on the uncaptured piece at G2: {notations:?}"
    );
    validate_board_actions(&board);
}

/// Different legal actions must not be reported with the same notation.
#[test]
fn distinct_actions_have_distinct_notations() {
    // Before the fix two different actions were both reported as a8xh8xh1xa1xa6xh6.
    let board = Board::from_squares(
        Team::White,
        &[Square::B3, Square::G5, Square::G6, Square::A8],
        &[
            Square::D1,
            Square::A2,
            Square::E6,
            Square::F6,
            Square::H7,
            Square::G8,
        ],
        &[Square::D1, Square::A8, Square::G8],
    );
    assert!(board.actions().len() > 1);
    validate_board_actions(&board);
}

/// A king capture loop with several kings on the board reports the king that
/// actually made the loop.
#[test]
fn king_loop_reports_the_correct_king() {
    //     A B C D E F G H
    //   +-----------------+
    // 8 | . . . . . . . . |
    // 7 | . . . W . . . . |  W = white kings at D7 and G6
    // 6 | . . . B . . W . |  B = black kings at D6 and E2
    // 5 | . . b . . . . . |  b = black man at C5
    // 4 | . . . . . . b . |  b = black man at G4
    // 3 | . . . . . . . . |
    // 2 | . . . . B . . . |
    // 1 | . . . . . . . . |
    //   +-----------------+
    //
    // Only the G6 king can loop: g6xc6xc2xg2xg6. The source must be G6, not D7.
    let board = Board::from_squares(
        Team::White,
        &[Square::G6, Square::D7],
        &[Square::E2, Square::G4, Square::C5, Square::D6],
        &[Square::E2, Square::D6, Square::G6, Square::D7],
    );
    let loops: Vec<_> = board
        .actions()
        .into_iter()
        .filter(|a| a.is_capture(Team::White) && a.delta.pieces[0] == 0)
        .collect();
    assert_eq!(loops.len(), 1, "expected exactly one king loop");
    let detailed = loops[0].to_detailed(Team::White, &board.state);
    assert_eq!(detailed.to_notation(), "g6xc6xc2xg2xg6");
    assert_eq!(detailed.source(), Square::G6);
    assert_eq!(detailed.destination(), Square::G6);
    validate_board_actions(&board);
}

/// Randomised check over many positions reached by deterministic random playouts:
/// every path is legal, matches its action, and no two actions are duplicates or
/// share a notation.
#[test]
fn random_playouts_have_legal_unique_paths() {
    // xorshift64: deterministic, no external dependency
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for _game in 0..300 {
        let mut board = Board::new_default();
        for _ply in 0..200 {
            validate_board_actions(&board);
            let actions = board.actions();
            if actions.is_empty() {
                break;
            }
            let pick = (next() % actions.len() as u64) as usize;
            board = board.apply(&actions[pick]).swap_turn();
        }
    }
}

// =============================================================================
// Helper functions
// =============================================================================

fn validate_all_actions_have_valid_paths(game: &Game) {
    validate_board_actions(game.board());
}

/// Checks every legal action of `board`:
/// - its path is geometrically valid (`validate_path`),
/// - replaying the path on the board obeys the capture rules (`replay_capture_path`)
///   and reaches exactly the position the action produces,
/// - no two actions produce the same position or share the same notation.
fn validate_board_actions(board: &Board) {
    let actions = board.actions();
    let mut notations = std::collections::HashSet::new();

    for (i, action) in actions.iter().enumerate() {
        let detailed = action.to_detailed(board.turn, &board.state);
        let path = detailed.path();
        let notation = detailed.to_notation();

        // Validate path is not empty
        assert!(!path.is_empty(), "Path should not be empty");

        // Validate path geometry
        if let Err(e) = validate_path(path) {
            panic!(
                "Invalid path for action {}: {}\nPath: {:?}",
                notation,
                e,
                path.iter().map(|s| s.to_string()).collect::<Vec<_>>()
            );
        }

        // Validate the path is legal on this board and matches the action's result
        if action.is_capture(board.turn) {
            let expected = board.apply(action).state;
            match replay_capture_path(board, path) {
                Ok((own, captured)) => {
                    assert_eq!(
                        own,
                        expected.pieces[board.turn.to_usize()],
                        "Path {notation} does not end where the action does"
                    );
                    assert_eq!(
                        captured,
                        action.captured_pieces(board.turn),
                        "Path {notation} does not capture the action's pieces"
                    );
                }
                Err(e) => panic!("Illegal path {notation}: {e}"),
            }
        }

        assert!(
            notations.insert(notation.clone()),
            "Two legal actions share the notation {notation}"
        );
        for other in &actions[i + 1..] {
            assert_ne!(
                action.delta, other.delta,
                "Duplicate action {notation} (same resulting position)"
            );
        }
    }
}

/// Replays a capture path square by square under the Turkish capture rules and
/// returns the mover's final pieces and the set of captured pieces.
///
/// Each step must be orthogonal and jump exactly one hostile piece: men jump an
/// adjacent piece forward or sideways, kings may fly over empty squares but not over
/// any other piece, nor reverse 180 degrees. Captured pieces are removed immediately
/// (Rule 6.4) and promotion only happens at the end of the turn (Rule 7.2).
fn replay_capture_path(board: &Board, path: &[Square]) -> Result<(u64, u64), String> {
    let team = board.turn.to_usize();
    let mut own = board.state.pieces[team];
    let mut hostile = board.state.pieces[1 - team];
    let is_king = board.state.kings & path[0].to_mask() != 0;
    let forward: i32 = if board.turn == Team::White { 1 } else { -1 };
    let mut captured = 0u64;
    let mut prev_dir: Option<(i32, i32)> = None;

    if own & path[0].to_mask() == 0 {
        return Err(format!("{} is not one of our pieces", path[0]));
    }

    for step in path.windows(2) {
        let (from, to) = (step[0], step[1]);
        let (r0, c0) = (i32::from(from.row()), i32::from(from.column()));
        let (r1, c1) = (i32::from(to.row()), i32::from(to.column()));
        if r0 != r1 && c0 != c1 {
            return Err(format!("{from}->{to} is not orthogonal"));
        }
        let dir = ((r1 - r0).signum(), (c1 - c0).signum());
        let dist = (r1 - r0).abs() + (c1 - c0).abs();
        if !is_king && (dir.0 == -forward || dist != 2) {
            return Err(format!("{from}->{to} is not a man's capture"));
        }
        if is_king && prev_dir == Some((-dir.0, -dir.1)) {
            return Err(format!("{from}->{to} reverses 180 degrees"));
        }

        let mut jumped = None;
        for k in 1..dist {
            let mask = 1u64 << ((r0 + dir.0 * k) * 8 + (c0 + dir.1 * k));
            if own & mask != 0 {
                return Err(format!("{from}->{to} jumps our own piece"));
            }
            if hostile & mask != 0 {
                if jumped.is_some() {
                    return Err(format!("{from}->{to} jumps two pieces"));
                }
                jumped = Some(mask);
            }
        }
        let jumped = jumped.ok_or_else(|| format!("{from}->{to} captures nothing"))?;
        if (own | hostile) & to.to_mask() != 0 {
            return Err(format!("{from}->{to} lands on an occupied square"));
        }

        hostile &= !jumped;
        captured |= jumped;
        own = (own & !from.to_mask()) | to.to_mask();
        prev_dir = Some(dir);
    }

    Ok((own, captured))
}
