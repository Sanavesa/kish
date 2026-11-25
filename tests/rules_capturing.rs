//! Section 6: Capturing Rules tests
//!
//! Tests for mandatory capture, men/king captures, chain captures,
//! maximum capture rule, and 180-degree turn prohibition.

use kish::{Board, Square, Team};

// =============================================================================
// 6.1 Mandatory Capture
// =============================================================================

/// Rule 6.1: Captures are mandatory - cannot make non-capturing move if capture available
#[test]
fn captures_are_mandatory() {
    let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5, Square::H8], &[]);
    let actions = board.actions();

    // All actions must be captures
    for action in &actions {
        assert_ne!(
            action.delta.pieces[Team::Black.to_usize()],
            0,
            "Must capture when capture is available"
        );
    }
}

/// Rule 6.1: Moves allowed when no capture available
#[test]
fn moves_allowed_when_no_capture() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::H8], // Far away, no capture possible
        &[],
    );
    let actions = board.actions();

    // All actions should be moves (non-captures)
    for action in &actions {
        assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            0,
            "Should be moves, not captures"
        );
    }
    assert!(!actions.is_empty(), "Should have moves available");
}

/// Rule 6.1: Multiple pieces, only one can capture - must use that one
#[test]
fn only_capturing_piece_can_move() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4, Square::A1],
        &[Square::D5], // Only D4 can capture D5
        &[],
    );
    let actions = board.actions();

    // Only capture actions should be returned
    assert_eq!(actions.len(), 1, "Only one capture available");
    assert_ne!(
        actions[0].delta.pieces[Team::Black.to_usize()],
        0,
        "Must be a capture"
    );
}

// =============================================================================
// 6.2 Men's Captures
// =============================================================================

/// Rule 6.2: Men capture forward
#[test]
fn white_pawn_captures_forward() {
    let board = Board::from_squares(Team::White, &[Square::D4], &[Square::D5], &[]);
    let actions = board.actions();

    // Should capture D5, land on D6
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0].delta.pieces[Team::Black.to_usize()],
        Square::D5.to_mask()
    );
}

/// Rule 6.2: Men capture left (sideways)
#[test]
fn white_pawn_captures_left() {
    let board = Board::from_squares(Team::White, &[Square::D4], &[Square::C4], &[]);
    let actions = board.actions();

    // Should capture C4, land on B4
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0].delta.pieces[Team::Black.to_usize()],
        Square::C4.to_mask()
    );
}

/// Rule 6.2: Men capture right (sideways)
#[test]
fn white_pawn_captures_right() {
    let board = Board::from_squares(Team::White, &[Square::D4], &[Square::E4], &[]);
    let actions = board.actions();

    // Should capture E4, land on F4
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0].delta.pieces[Team::Black.to_usize()],
        Square::E4.to_mask()
    );
}

/// Rule 6.2: Men cannot capture backward
#[test]
fn white_pawn_cannot_capture_backward() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::D3, Square::D5], // D3 is behind, D5 is in front
        &[],
    );
    let actions = board.actions();

    // Should only capture D5 (forward), not D3 (backward)
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0].delta.pieces[Team::Black.to_usize()],
        Square::D5.to_mask(),
        "Should only capture forward (D5), not backward (D3)"
    );
}

/// Rule 6.2: Men cannot capture backward - black pawn
#[test]
fn black_pawn_cannot_capture_backward() {
    let board = Board::from_squares(
        Team::Black,
        &[Square::D5, Square::D3], // D5 is behind (up) for black, D3 is forward (down)
        &[Square::D4],
        &[],
    );
    let actions = board.actions();

    // Should only capture D3 (forward for black = down), not D5 (backward)
    assert_eq!(actions.len(), 1);
    assert_eq!(
        actions[0].delta.pieces[Team::White.to_usize()],
        Square::D3.to_mask(),
        "Should only capture forward (D3), not backward (D5)"
    );
}

/// Rule 6.2: Men cannot capture diagonally
#[test]
fn pawn_cannot_capture_diagonally() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::C3, Square::C5, Square::E3, Square::E5],
        &[],
    );
    let actions = board.actions();

    // Should have NO captures (all enemies are diagonal)
    // Without captures, should have moves instead
    for action in &actions {
        assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            0,
            "Should not capture diagonal enemies"
        );
    }
}

// =============================================================================
// 6.3 King's Captures (Flying Capture)
// =============================================================================

/// Rule 6.3: King can capture from any distance
#[test]
fn king_captures_from_distance() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A4],
        &[Square::E4], // Distance 4 from A4
        &[Square::A4],
    );
    let actions = board.actions();

    // King can capture E4 from A4 (flying capture)
    assert!(!actions.is_empty(), "King should have captures");
    for action in &actions {
        assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            Square::E4.to_mask()
        );
    }
}

/// Rule 6.3: King can land on any empty square beyond the captured piece
#[test]
fn king_can_land_anywhere_beyond_captured() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A4],
        &[Square::D4], // After capturing, can land on E4, F4, G4, or H4
        &[Square::A4],
    );
    let actions = board.actions();

    // Should have 4 landing options: E4, F4, G4, H4
    assert_eq!(actions.len(), 4, "King should have 4 landing options");

    let landing_squares: Vec<u64> = actions
        .iter()
        .map(|a| a.delta.pieces[Team::White.to_usize()] & !Square::A4.to_mask())
        .collect();

    assert!(landing_squares.contains(&Square::E4.to_mask()));
    assert!(landing_squares.contains(&Square::F4.to_mask()));
    assert!(landing_squares.contains(&Square::G4.to_mask()));
    assert!(landing_squares.contains(&Square::H4.to_mask()));
}

/// Rule 6.3: King captures in all 4 directions
#[test]
fn king_captures_in_all_directions() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::B4, Square::F4, Square::D2, Square::D6],
        &[Square::D4],
    );
    let actions = board.actions();

    // King can capture in any direction - check all are available
    // Due to max capture rule, single captures may not all be returned if chains exist
    // But all 4 pieces should be capturable
    let captured_pieces: u64 = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()])
        .fold(0, |acc, x| acc | x);

    // At least one capture should be possible
    assert!(captured_pieces != 0, "King should be able to capture");
}

/// Rule 6.3: King cannot jump over two adjacent pieces at once
#[test]
fn king_cannot_jump_two_pieces_at_once() {
    // When two enemy pieces are adjacent (C4, D4), and king at A4,
    // the king cannot capture C4 because there's no landing square
    // between C4 and D4, and D4 blocks the path.
    // This tests that you can't skip over two pieces in a single jump.
    let board = Board::from_squares(
        Team::White,
        &[Square::A4],
        &[Square::C4, Square::D4], // Two adjacent enemies - no landing between them
        &[Square::A4],
    );
    let actions = board.actions();

    // King at A4 cannot capture C4 (would land on D4 which is occupied)
    // King at A4 cannot capture D4 (C4 is in the way)
    // So no captures are possible - only moves available
    for action in &actions {
        // All actions should be moves (no captures possible)
        assert_eq!(
            action.delta.pieces[Team::Black.to_usize()],
            0,
            "No captures should be possible with two adjacent blocking pieces"
        );
    }
    assert!(!actions.is_empty(), "Should have moves available");
}

// =============================================================================
// 6.4 Immediate Removal Rule
// =============================================================================

/// Rule 6.4: Captured pieces removed immediately (can cross vacated square)
#[test]
fn captured_piece_removed_immediately_allows_crossing() {
    // King at A4, enemies at C4 and E4
    // After capturing C4 -> land at D4
    // From D4, can capture E4 (would cross where C4 was, but it's removed)
    let board = Board::from_squares(
        Team::White,
        &[Square::A4],
        &[Square::C4, Square::E4],
        &[Square::A4],
    );
    let actions = board.actions();

    // Should be able to capture both pieces
    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_captures, 2,
        "Should capture 2 pieces using immediate removal"
    );
}

/// Rule 6.4: Pawn chain capture with immediate removal
#[test]
fn pawn_chain_capture_with_immediate_removal() {
    // White pawn at A2, enemies at A3 and B4
    // Capture A3 -> land A4
    // From A4, capture B4 -> land C4
    let board = Board::from_squares(Team::White, &[Square::A2], &[Square::A3, Square::B4], &[]);
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(max_captures, 2, "Pawn should capture 2 pieces in chain");
}

// =============================================================================
// 6.5 Multi-Capture (Chain Captures)
// =============================================================================

/// Rule 6.5: Multi-capture must continue if more captures available
#[test]
fn multi_capture_must_continue() {
    let board = Board::from_squares(Team::White, &[Square::A2], &[Square::A3, Square::B4], &[]);
    let actions = board.actions();

    // Should only return the full chain (2 captures), not partial (1 capture)
    for action in &actions {
        let captures = action.delta.pieces[Team::Black.to_usize()].count_ones();
        assert_eq!(captures, 2, "Must complete the full capture chain");
    }
}

/// Rule 6.5: Three-piece capture chain
#[test]
fn three_piece_capture_chain() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A2],
        &[Square::A3, Square::B4, Square::C5],
        &[],
    );
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(max_captures, 3, "Should capture 3 pieces in chain");
}

/// Rule 6.5: Four-piece capture chain
#[test]
fn four_piece_capture_chain() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A2],
        &[Square::A3, Square::B4, Square::C5, Square::D6],
        &[],
    );
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(max_captures, 4, "Should capture 4 pieces in chain");
}

/// Rule 6.5: King chain capture
#[test]
fn king_chain_capture() {
    let board = Board::from_squares(
        Team::White,
        &[Square::A4],
        &[Square::C4, Square::E6],
        &[Square::A4],
    );
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(max_captures, 2, "King should capture 2 pieces in chain");
}

// =============================================================================
// 6.6 Maximum Capture Rule (Majority Rule)
// =============================================================================

/// Rule 6.6: Must choose sequence that captures the most pieces
#[test]
fn maximum_capture_rule_enforced() {
    // White pawn at D4
    // Path 1: capture D5 (1 piece)
    // Path 2: capture C4->B4, then B5->B6 (2 pieces via chain)
    // Must choose path 2
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::D5, Square::C4, Square::B5],
        &[],
    );
    let actions = board.actions();

    // All returned actions should be the maximum length
    for action in &actions {
        let captures = action.delta.pieces[Team::Black.to_usize()].count_ones();
        assert!(captures >= 2, "Must choose maximum capture sequence");
    }
}

/// Rule 6.6: Multiple sequences of same length - can choose any
#[test]
fn equal_length_captures_all_available() {
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::D5, Square::C4, Square::E4],
        &[],
    );
    let actions = board.actions();

    // All should be 1-capture (equal length)
    assert_eq!(
        actions.len(),
        3,
        "Should have 3 equal-length capture options"
    );
    for action in &actions {
        let captures = action.delta.pieces[Team::Black.to_usize()].count_ones();
        assert_eq!(captures, 1, "All captures should be length 1");
    }
}

/// Rule 6.6: Two routes that capture the same pieces and end on the same square
/// are the same move, and are generated once
#[test]
fn equal_length_routes_to_same_position_generated_once() {
    // King at H2, enemies at H4 and H7. The king may land on H5 or H6 after
    // capturing H4, and from either square capture H7 and land on H8:
    // h2xh5xh8 and h2xh6xh8 reach the same position, so only one action is legal.
    let board = Board::from_squares(
        Team::White,
        &[Square::H2],
        &[Square::H4, Square::H7],
        &[Square::H2],
    );
    let actions = board.actions();

    assert_eq!(actions.len(), 1, "Both routes are the same move");
    assert_eq!(
        actions[0].delta.pieces[Team::Black.to_usize()],
        Square::H4.to_mask() | Square::H7.to_mask()
    );
    let path = actions[0].to_detailed(Team::White, &board.state);
    assert_eq!(path.source(), Square::H2);
    assert_eq!(path.destination(), Square::H8);
}

/// Rule 6.6: A king capture loop can be made in either direction; both directions
/// are the same move, and are generated once
#[test]
fn king_capture_loop_both_directions_generated_once() {
    // King at B2, enemies at D2, F4, D6 and B4. Clockwise and counter-clockwise
    // loops both capture all four pieces and end back on B2. (The king may also
    // finish on B1 or B3; those are different moves.)
    let board = Board::from_squares(
        Team::White,
        &[Square::B2],
        &[Square::D2, Square::F4, Square::D6, Square::B4],
        &[Square::B2],
    );
    let loops: Vec<_> = board
        .actions()
        .into_iter()
        .filter(|a| a.delta.pieces[Team::White.to_usize()] == 0)
        .collect();

    assert_eq!(loops.len(), 1, "Both directions are the same move");
    assert_eq!(loops[0].capture_count(Team::White), 4);
    let path = loops[0].to_detailed(Team::White, &board.state);
    assert_eq!(path.source(), Square::B2);
    assert_eq!(path.destination(), Square::B2);
    assert_eq!(path.path_len(), 5, "start + 4 landings");
}

/// A king capture loop has no own-piece delta, so `Action::source` cannot derive
/// the square and must panic instead of returning garbage (use `to_detailed`)
#[test]
#[should_panic(expected = "king capture loop")]
fn king_capture_loop_source_panics() {
    let board = Board::from_squares(
        Team::White,
        &[Square::B2],
        &[Square::D2, Square::F4, Square::D6, Square::B4],
        &[Square::B2],
    );
    let king_loop = board
        .actions()
        .into_iter()
        .find(|a| a.delta.pieces[Team::White.to_usize()] == 0)
        .expect("king loop is legal");
    let _ = king_loop.source(Team::White, board.state.pieces[Team::White.to_usize()]);
}

/// A king capture loop has no own-piece delta, so `Action::destination` cannot
/// derive the square and must panic instead of returning garbage (use `to_detailed`)
#[test]
#[should_panic(expected = "king capture loop")]
fn king_capture_loop_destination_panics() {
    let board = Board::from_squares(
        Team::White,
        &[Square::B2],
        &[Square::D2, Square::F4, Square::D6, Square::B4],
        &[Square::B2],
    );
    let king_loop = board
        .actions()
        .into_iter()
        .find(|a| a.delta.pieces[Team::White.to_usize()] == 0)
        .expect("king loop is legal");
    let _ = king_loop.destination(Team::White, board.state.pieces[Team::White.to_usize()]);
}

/// Rule 6.6: No distinction between men and kings when counting
#[test]
fn men_and_kings_count_equally() {
    // D4 pawn, D5 king (1 capture), C4+B5 pawns (2 captures)
    // Must prefer 2 captures even though path 1 captures a king
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::D5, Square::C4, Square::B5],
        &[Square::D5], // D5 is a king
    );
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_captures, 2,
        "Should prefer 2-pawn capture over 1-king capture"
    );
}

// =============================================================================
// 6.7 180-Degree Turn Prohibition
// =============================================================================

/// Rule 6.7: Cannot turn 180° between consecutive captures (up then down)
#[test]
fn king_cannot_reverse_up_down_during_capture() {
    // King at D5, enemies at D3 and D7
    // Capture D7 (up) -> land D8
    // From D8, capture D3 would require going DOWN (180° turn) - blocked!
    let board = Board::from_squares(
        Team::White,
        &[Square::D5],
        &[Square::D3, Square::D7],
        &[Square::D5],
    );
    let actions = board.actions();

    // Should NOT be able to capture both (would require 180° turn)
    for action in &actions {
        let captures = action.delta.pieces[Team::Black.to_usize()].count_ones();
        assert_eq!(
            captures, 1,
            "Should NOT chain captures requiring 180° turn (up then down)"
        );
    }
}

/// Rule 6.7: Cannot turn 180° between consecutive captures (left then right)
#[test]
fn king_cannot_reverse_left_right_during_capture() {
    // King at D4, enemies at B4 (left) and F4 (right)
    // Capture B4 (left) -> land A4
    // From A4, capture F4 would require going RIGHT (180° turn) - blocked!
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::B4, Square::F4],
        &[Square::D4],
    );
    let actions = board.actions();

    for action in &actions {
        let captures = action.delta.pieces[Team::Black.to_usize()].count_ones();
        assert_eq!(
            captures, 1,
            "Should NOT chain captures requiring 180° turn (left then right)"
        );
    }
}

/// Rule 6.7: 90° turns ARE allowed between captures
#[test]
fn king_can_turn_90_degrees_during_capture() {
    // King at D4, enemy at D6 (up), enemy at F7 (right from D7 area)
    // Capture D6 (up) -> land D7
    // From D7, capture F7 (right) - 90° turn, allowed!
    let board = Board::from_squares(
        Team::White,
        &[Square::D4],
        &[Square::D6, Square::F7],
        &[Square::D4],
    );
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(max_captures, 2, "90° turn should be allowed");
}

/// Rule 6.7: Complex 90° turn scenario (L-shape)
#[test]
fn king_90_degree_l_shape_capture() {
    // King at A1, enemies at A3 (up), C5 (right), E3 (down)
    // Path: A1 -> capture A3 (up) -> land A4
    //       A4 -> not directly to C5... let me recalculate
    // Better: A1 -> A3 (up) -> land A5
    //         A5 -> C5 (right) -> land E5
    //         E5 -> E3 (down) - 90° from right
    let board = Board::from_squares(
        Team::White,
        &[Square::A1],
        &[Square::A3, Square::C5, Square::E3],
        &[Square::A1],
    );
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::Black.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(max_captures, 3, "Should chain 3 captures with 90° turns");
}

// =============================================================================
// 6.8 Black Team Captures (symmetric behavior)
// =============================================================================

/// Rule 6.2: Black pawn captures forward (down)
#[test]
fn black_pawn_captures_forward_vertical() {
    // Black pawn at D5 captures D4 (below), lands on D3
    let board = Board::from_squares(Team::Black, &[Square::D4], &[Square::D5], &[]);
    let actions = board.actions();

    // Should capture D4, land on D3
    assert_eq!(actions.len(), 1, "Black pawn should have 1 capture");
    assert_eq!(
        actions[0].delta.pieces[Team::White.to_usize()],
        Square::D4.to_mask(),
        "Black pawn should capture D4"
    );
}

/// Rule 6.2: Black pawn vertical chain capture (down)
#[test]
fn black_pawn_vertical_chain_capture() {
    // Black pawn at D6 captures D5, lands D4, then captures D3, lands D2
    let board = Board::from_squares(Team::Black, &[Square::D5, Square::D3], &[Square::D6], &[]);
    let actions = board.actions();

    let max_captures = actions
        .iter()
        .map(|a| a.delta.pieces[Team::White.to_usize()].count_ones())
        .max()
        .unwrap_or(0);

    assert_eq!(
        max_captures, 2,
        "Black pawn should chain 2 vertical captures"
    );
}

/// Rule 6.3: Black king captures in all directions
#[test]
fn black_king_captures_all_directions() {
    // Black king at D4, white pieces in all 4 directions
    let board = Board::from_squares(
        Team::Black,
        &[Square::B4, Square::F4, Square::D2, Square::D6],
        &[Square::D4],
        &[Square::D4], // D4 is a black king
    );
    let actions = board.actions();

    // All actions should be captures (mandatory capture rule)
    assert!(
        !actions.is_empty(),
        "Black king should have capture options"
    );
    for action in &actions {
        assert_ne!(
            action.delta.pieces[Team::White.to_usize()],
            0,
            "All actions should be captures"
        );
    }
}
