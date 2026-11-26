"""Tests for the Square enum."""

import pytest
import kish


def test_square_values():
    """Test Square enum has correct values."""
    assert int(kish.Square.A1) == 0
    assert int(kish.Square.H1) == 7
    assert int(kish.Square.A8) == 56
    assert int(kish.Square.H8) == 63


def test_square_from_notation():
    """Test Square.from_notation() parsing."""
    assert kish.Square.from_notation("a1") == kish.Square.A1
    assert kish.Square.from_notation("A1") == kish.Square.A1
    assert kish.Square.from_notation("h8") == kish.Square.H8
    assert kish.Square.from_notation("H8") == kish.Square.H8
    assert kish.Square.from_notation("d4") == kish.Square.D4


def test_square_from_notation_invalid():
    """Test Square.from_notation() raises on invalid input."""
    with pytest.raises(ValueError):
        kish.Square.from_notation("")
    with pytest.raises(ValueError):
        kish.Square.from_notation("a")
    with pytest.raises(ValueError):
        kish.Square.from_notation("a1b")
    with pytest.raises(ValueError):
        kish.Square.from_notation("i1")  # Invalid file
    with pytest.raises(ValueError):
        kish.Square.from_notation("a9")  # Invalid rank


def test_square_from_row_col():
    """Test Square.from_row_col()."""
    assert kish.Square.from_row_col(0, 0) == kish.Square.A1
    assert kish.Square.from_row_col(0, 7) == kish.Square.H1
    assert kish.Square.from_row_col(7, 0) == kish.Square.A8
    assert kish.Square.from_row_col(7, 7) == kish.Square.H8
    assert kish.Square.from_row_col(3, 3) == kish.Square.D4


def test_square_from_row_col_invalid():
    """Test Square.from_row_col() raises on invalid input."""
    with pytest.raises(ValueError):
        kish.Square.from_row_col(8, 0)
    with pytest.raises(ValueError):
        kish.Square.from_row_col(0, 8)


def test_square_row_col():
    """Test Square.row() and Square.col()."""
    assert kish.Square.A1.row() == 0
    assert kish.Square.A1.col() == 0
    assert kish.Square.H8.row() == 7
    assert kish.Square.H8.col() == 7
    assert kish.Square.D4.row() == 3
    assert kish.Square.D4.col() == 3


def test_square_notation():
    """Test Square.notation()."""
    assert kish.Square.A1.notation() == "A1"
    assert kish.Square.H8.notation() == "H8"
    assert kish.Square.D4.notation() == "D4"


def test_square_manhattan():
    """Test Square.manhattan() distance."""
    assert kish.Square.A1.manhattan(kish.Square.A1) == 0
    assert kish.Square.A1.manhattan(kish.Square.A2) == 1
    assert kish.Square.A1.manhattan(kish.Square.B1) == 1
    assert kish.Square.A1.manhattan(kish.Square.H8) == 14
    assert kish.Square.D4.manhattan(kish.Square.D4) == 0
    assert kish.Square.D4.manhattan(kish.Square.H8) == 8


def test_square_to_mask():
    """Test Square.to_mask() returns correct bitboard."""
    assert kish.Square.A1.to_mask() == 1 << 0
    assert kish.Square.H1.to_mask() == 1 << 7
    assert kish.Square.A8.to_mask() == 1 << 56
    assert kish.Square.H8.to_mask() == 1 << 63


def test_square_from_mask():
    """Test Square.from_mask() roundtrip."""
    for sq in [kish.Square.A1, kish.Square.D4, kish.Square.H8]:
        mask = sq.to_mask()
        assert kish.Square.from_mask(mask) == sq


def test_square_from_mask_invalid():
    """Test Square.from_mask() raises on invalid mask."""
    with pytest.raises(ValueError):
        kish.Square.from_mask(0)  # No bits set
    with pytest.raises(ValueError):
        kish.Square.from_mask(3)  # Multiple bits set


def test_square_roundtrip_through_engine():
    """Test every square converts to and from the engine unchanged."""
    for index in range(64):
        square = kish.Square.from_mask(1 << index)
        assert square.to_mask() == 1 << index
        assert square.row() * 8 + square.col() == index
        board = kish.Board.from_squares(
            turn=kish.Team.White,
            white_squares=[square],
            black_squares=[],
            king_squares=[square],
        )
        assert board.white_pieces() == [square]


def test_square_hash():
    """Test Square is hashable."""
    squares = {kish.Square.A1, kish.Square.H8}
    assert len(squares) == 2
    assert kish.Square.A1 in squares
