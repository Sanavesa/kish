"""Tests for the Action class."""

import kish


def test_action_source_destination():
    """Test Action.source() and Action.destination()."""
    board = kish.Board()
    actions = board.actions()
    action = actions[0]

    src = action.source()
    dst = action.destination()

    assert isinstance(src, kish.Square)
    assert isinstance(dst, kish.Square)
    assert src != dst


def test_action_notation():
    """Test Action.notation() format."""
    board = kish.Board()
    actions = board.actions()

    for action in actions:
        notation = action.notation()
        # Non-capture moves have format "a1-a2"
        assert "-" in notation or "x" in notation
        assert len(notation) >= 5  # At least "a1-a2"


def test_action_is_capture():
    """Test Action.is_capture() for non-capture moves."""
    board = kish.Board()
    actions = board.actions()

    # Starting position has no captures
    for action in actions:
        assert not action.is_capture()


def test_action_is_capture_true():
    """Test Action.is_capture() for capture moves."""
    # Position where white must capture
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.D5, kish.Square.H8],
        king_squares=[],
    )
    actions = board.actions()

    # Mandatory capture rule
    assert len(actions) == 1
    assert actions[0].is_capture()
    assert "x" in actions[0].notation()


def test_action_captured_pieces():
    """Test Action.captured_pieces() returns captured squares."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.D5, kish.Square.H8],
        king_squares=[],
    )
    actions = board.actions()
    action = actions[0]

    captured = action.captured_pieces()
    assert len(captured) == 1
    assert kish.Square.D5 in captured


def test_action_captured_bitboard():
    """Test Action.captured_bitboard() returns u64."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.D5, kish.Square.H8],
        king_squares=[],
    )
    actions = board.actions()
    action = actions[0]

    captured_bb = action.captured_bitboard()
    assert captured_bb == kish.Square.D5.to_mask()


def test_action_capture_count():
    """Test Action.capture_count()."""
    board = kish.Board()
    actions = board.actions()

    # No captures in starting position
    for action in actions:
        assert action.capture_count() == 0


def test_action_path():
    """Test Action.path() returns square list."""
    board = kish.Board()
    actions = board.actions()
    action = actions[0]

    path = action.path()
    assert len(path) >= 2
    assert path[0] == action.source()
    assert path[-1] == action.destination()


def test_action_delta():
    """Test Action.delta() returns bitboard tuple."""
    board = kish.Board()
    actions = board.actions()
    action = actions[0]

    white_delta, black_delta, kings_delta = action.delta()

    # For a simple white move, white_delta should have 2 bits set (src and dst)
    assert bin(white_delta).count("1") == 2  # int.bit_count() needs Python 3.10+
    assert black_delta == 0  # No black pieces affected
    assert kings_delta == 0  # No kings in starting position


def test_action_delta_array():
    """Test Action.delta_array() returns array."""
    board = kish.Board()
    actions = board.actions()
    action = actions[0]

    arr = action.delta_array()
    assert len(arr) == 3
    assert arr == list(action.delta())


def test_action_is_promotion():
    """Test Action.is_promotion() for promotion moves."""
    # White pawn about to promote
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D7],
        black_squares=[kish.Square.A2],
        king_squares=[],
    )
    actions = board.actions()

    # D7-D8 should promote
    promotion_action = next(a for a in actions if a.destination() == kish.Square.D8)
    assert promotion_action.is_promotion()
    assert "=K" in promotion_action.notation()


def test_action_hash():
    """Test Action is hashable."""
    board = kish.Board()
    actions = board.actions()
    action_set = set(actions)
    assert len(action_set) == len(actions)


def test_action_equality():
    """Test Action equality."""
    board = kish.Board()
    actions1 = board.actions()
    actions2 = board.actions()

    # Same actions from same position should be equal
    assert actions1[0] == actions2[0]
