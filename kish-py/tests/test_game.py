"""Tests for the Game class."""

import pytest

import kish


def test_game_new():
    """Test Game() creates new game."""
    game = kish.Game()
    assert game.turn == kish.Team.White
    assert game.move_count == 0
    assert game.halfmove_clock == 0


def test_game_from_board():
    """Test Game.from_board() creates game from position."""
    board = kish.Board.from_squares(
        turn=kish.Team.Black,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.E5],
        king_squares=[],
    )
    game = kish.Game.from_board(board)
    assert game.turn == kish.Team.Black


def test_game_board():
    """Test Game.board() returns current board."""
    game = kish.Game()
    board = game.board()
    assert isinstance(board, kish.Board)
    assert board.turn == kish.Team.White


def test_game_actions():
    """Test Game.actions() returns legal moves."""
    game = kish.Game()
    actions = game.actions()
    assert len(actions) > 0


def test_game_make_move():
    """Test Game.make_move() updates state."""
    game = kish.Game()
    actions = game.actions()

    game.make_move(actions[0])

    assert game.turn == kish.Team.Black
    assert game.move_count == 1


def test_game_make_move_rejects_illegal_action():
    """Test Game.make_move() raises ValueError and leaves the game unchanged."""
    game = kish.Game()
    action = game.actions()[0]
    game.make_move(action)
    board = game.board()

    # Playing the same action again used to revert it and corrupt the history
    with pytest.raises(ValueError, match="illegal action"):
        game.make_move(action)

    assert game.board() == board
    assert game.move_count == 1
    assert game.halfmove_clock == 1


def test_game_undo_move():
    """Test Game.undo_move() restores state."""
    game = kish.Game()
    original_board = game.board()

    actions = game.actions()
    game.make_move(actions[0])
    assert game.move_count == 1

    success = game.undo_move()
    assert success
    assert game.move_count == 0
    assert game.board() == original_board


def test_game_undo_move_empty():
    """Test Game.undo_move() returns False when empty."""
    game = kish.Game()
    assert not game.undo_move()


def test_game_halfmove_clock():
    """Test halfmove_clock increments for non-captures."""
    # Position with only kings (no captures possible without setup)
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.A1, kish.Square.B1],
        black_squares=[kish.Square.H7, kish.Square.H8],
        king_squares=[kish.Square.A1, kish.Square.B1, kish.Square.H7, kish.Square.H8],
    )
    game = kish.Game.from_board(board)

    actions = game.actions()
    game.make_move(actions[0])

    assert game.halfmove_clock == 1


def test_game_halfmove_clock_reset():
    """Test halfmove_clock resets on capture."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.D5, kish.Square.H8],
        king_squares=[],
    )
    game = kish.Game.from_board(board)

    # Manually set clock to non-zero (simulating previous non-capture moves)
    # Note: We can't directly set it, so we just verify it resets after capture
    actions = game.actions()
    game.make_move(actions[0])  # Capture

    assert game.halfmove_clock == 0


def test_game_status():
    """Test Game.status() returns game status."""
    game = kish.Game()
    status = game.status()
    assert status.is_in_progress()


def test_game_position_count():
    """Test Game.position_count() tracks occurrences."""
    game = kish.Game()
    assert game.position_count() == 1  # Initial position

    # Make and undo a move (returns to same position)
    actions = game.actions()
    game.make_move(actions[0])
    game.undo_move()

    assert game.position_count() == 1  # Should be decremented on undo


def test_game_is_threefold_repetition():
    """Test Game.is_threefold_repetition()."""
    game = kish.Game()
    assert not game.is_threefold_repetition()


def test_game_clear_history():
    """Test Game.clear_history() resets tracking."""
    game = kish.Game()
    actions = game.actions()
    game.make_move(actions[0])

    game.clear_history()

    assert game.move_count == 0
    assert game.halfmove_clock == 0
    assert game.position_count() == 1


def test_game_perft():
    """Test Game.perft() matches Board.perft()."""
    game = kish.Game()
    board = kish.Board()

    for depth in range(4):
        assert game.perft(depth) == board.perft(depth)
