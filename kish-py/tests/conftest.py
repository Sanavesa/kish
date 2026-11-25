"""Pytest configuration and fixtures."""

import pytest
import kish


@pytest.fixture
def default_board():
    """Return a board with the standard starting position."""
    return kish.Board()


@pytest.fixture
def capture_position():
    """Return a position where white must capture."""
    return kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.D5, kish.Square.H8],
        king_squares=[],
    )


@pytest.fixture
def promotion_position():
    """Return a position where white can promote."""
    return kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D7],
        black_squares=[kish.Square.A2],
        king_squares=[],
    )


@pytest.fixture
def draw_position():
    """Return a 1v1 draw position."""
    return kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.A1],
        black_squares=[kish.Square.H8],
        king_squares=[],
    )


@pytest.fixture
def king_position():
    """Return a position with kings."""
    return kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.D4],
        black_squares=[kish.Square.E6],
        king_squares=[kish.Square.D4, kish.Square.E6],
    )
