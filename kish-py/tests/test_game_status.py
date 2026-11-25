"""Tests for the GameStatus class."""

import kish


def test_game_status_in_progress():
    """Test GameStatus for in-progress game."""
    board = kish.Board()
    status = board.status()

    assert status.is_in_progress()
    assert not status.is_draw()
    assert not status.is_won()
    assert not status.is_over()
    assert status.winner() is None


def test_game_status_draw():
    """Test GameStatus for draw (1v1)."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.A1],
        black_squares=[kish.Square.H8],
        king_squares=[],
    )
    status = board.status()

    assert not status.is_in_progress()
    assert status.is_draw()
    assert not status.is_won()
    assert status.is_over()
    assert status.winner() is None


def test_game_status_won_white():
    """Test GameStatus when white wins."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.A1, kish.Square.B1],
        black_squares=[],  # No black pieces
        king_squares=[],
    )
    status = board.status()

    assert not status.is_in_progress()
    assert not status.is_draw()
    assert status.is_won()
    assert status.is_over()
    assert status.winner() == kish.Team.White


def test_game_status_won_black():
    """Test GameStatus when black wins."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[],  # No white pieces
        black_squares=[kish.Square.A2, kish.Square.B2],
        king_squares=[],
    )
    status = board.status()

    assert status.is_won()
    assert status.winner() == kish.Team.Black


def test_game_status_str():
    """Test GameStatus string representation."""
    board = kish.Board()
    status = board.status()
    assert "Progress" in str(status)


def test_game_status_repr():
    """Test GameStatus repr."""
    board = kish.Board()
    status = board.status()
    assert "GameStatus" in repr(status)


def test_game_status_equality():
    """Test GameStatus equality."""
    board1 = kish.Board()
    board2 = kish.Board()
    assert board1.status() == board2.status()


def test_game_status_hash():
    """Test GameStatus is hashable."""
    board = kish.Board()
    status = board.status()
    # Should not raise
    hash(status)
