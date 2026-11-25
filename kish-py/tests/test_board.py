"""Tests for the Board class."""

import pytest

import kish


def test_board_new():
    """Test Board() creates standard starting position."""
    board = kish.Board()
    assert board.turn == kish.Team.White
    assert len(board.white_pieces()) == 16
    assert len(board.black_pieces()) == 16
    assert len(board.kings()) == 0


def test_board_from_squares():
    """Test Board.from_squares() creates custom position."""
    board = kish.Board.from_squares(
        turn=kish.Team.Black,
        white_squares=[kish.Square.D4, kish.Square.E3],
        black_squares=[kish.Square.D5],
        king_squares=[kish.Square.D4],
    )
    assert board.turn == kish.Team.Black
    assert len(board.white_pieces()) == 2
    assert len(board.black_pieces()) == 1
    assert len(board.kings()) == 1
    assert kish.Square.D4 in board.kings()


def test_board_from_bitboards():
    """Test Board.from_bitboards() creates position from raw bitboards."""
    # Row 2 for white, row 7 for black
    white_bb = 0x0000_0000_0000_FF00
    black_bb = 0x00FF_0000_0000_0000

    board = kish.Board.from_bitboards(
        turn=0,  # White
        white=white_bb,
        black=black_bb,
        kings=0,
    )
    assert board.turn == kish.Team.White
    assert board.white_bitboard() == white_bb
    assert board.black_bitboard() == black_bb


D4, E5, A1, H8, D8 = 1 << 27, 1 << 36, 1 << 0, 1 << 63, 1 << 59


@pytest.mark.parametrize(
    "white, black, kings",
    [
        (D4, D4 | E5, 0),  # a square holds both colours
        (D4, E5, D4 | (1 << 40)),  # king flag on an empty square
        ((1 << 17) - 1, H8, 0),  # 17 white pieces
        (D4, (1 << 17) - 1 << 40, 0),  # 17 black pieces
        (D8, E5, 0),  # white man on its promotion row
        (D4, A1, 0),  # black man on its promotion row
    ],
)
def test_board_from_bitboards_rejects_invalid_positions(white, black, kings):
    """Test Board.from_bitboards() raises ValueError for invalid positions."""
    with pytest.raises(ValueError, match="invalid position"):
        kish.Board.from_bitboards(turn=0, white=white, black=black, kings=kings)


def test_board_from_squares_rejects_invalid_positions():
    """Test Board.from_squares() raises ValueError for invalid positions."""
    with pytest.raises(ValueError, match="both White and Black"):
        kish.Board.from_squares(
            turn=kish.Team.White,
            white_squares=[kish.Square.D4],
            black_squares=[kish.Square.D4],
            king_squares=[],
        )
    with pytest.raises(ValueError, match="subset"):
        kish.Board.from_squares(
            turn=kish.Team.White,
            white_squares=[kish.Square.D4],
            black_squares=[kish.Square.E5],
            king_squares=[kish.Square.F6],
        )


def test_board_from_bitboards_accepts_kings_on_promotion_rows():
    """Test kings (unlike men) may stand on any row, and a side may be empty."""
    board = kish.Board.from_bitboards(turn=0, white=D8, black=A1, kings=D8 | A1)
    assert len(board.kings()) == 2
    board = kish.Board.from_bitboards(turn=0, white=D4, black=0, kings=0)
    assert board.status().winner() == kish.Team.White


def test_board_actions():
    """Test Board.actions() returns legal moves."""
    board = kish.Board()
    actions = board.actions()
    # Standard position has 23 legal moves for white
    # (8 pawns * 3 directions - some blocked by edges)
    assert len(actions) > 0
    assert all(isinstance(a, kish.Action) for a in actions)


def test_board_apply():
    """Test Board.apply() returns new board."""
    board = kish.Board()
    actions = board.actions()
    new_board = board.apply(actions[0])

    # Original unchanged
    assert board.turn == kish.Team.White

    # New board has swapped turn
    assert new_board.turn == kish.Team.Black


def test_board_status():
    """Test Board.status() returns game status."""
    board = kish.Board()
    status = board.status()
    assert status.is_in_progress()
    assert not status.is_over()


def test_board_status_draw():
    """Test 1v1 is a draw."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.A1],
        black_squares=[kish.Square.H8],
        king_squares=[],
    )
    status = board.status()
    assert status.is_draw()
    assert status.is_over()


def test_board_bitboards():
    """Test Board.bitboards() returns tuple."""
    board = kish.Board()
    white, black, kings, turn = board.bitboards()
    assert white == board.white_bitboard()
    assert black == board.black_bitboard()
    assert kings == board.kings_bitboard()
    assert turn == 0  # White


def test_board_to_array():
    """Test Board.to_array() returns array."""
    board = kish.Board()
    arr = board.to_array()
    assert len(arr) == 4
    assert arr[0] == board.white_bitboard()
    assert arr[1] == board.black_bitboard()
    assert arr[2] == board.kings_bitboard()
    assert arr[3] == 0  # White turn


def test_board_rotate():
    """Test Board.rotate() rotates 180 degrees and swaps teams."""
    board = kish.Board.from_squares(
        turn=kish.Team.White,
        white_squares=[kish.Square.B3],
        black_squares=[kish.Square.F5],
        king_squares=[],
    )
    rotated = board.rotate()
    # After rotation, pieces rotate 180 degrees AND swap colors:
    # B3 (row 2, col 1) -> G6 (row 5, col 6), and becomes black
    # F5 (row 4, col 5) -> C4 (row 3, col 2), and becomes white
    assert kish.Square.C4 in rotated.white_pieces()
    assert kish.Square.G6 in rotated.black_pieces()


def test_board_perft_depth_0():
    """Test perft at depth 0."""
    board = kish.Board()
    assert board.perft(0) == 1


def test_board_perft_depth_1():
    """Test perft at depth 1."""
    board = kish.Board()
    # Turkish Draughts starting position: 8 legal moves (front row pawns)
    assert board.perft(1) == 8


def test_board_perft_depth_2():
    """Test perft at depth 2."""
    board = kish.Board()
    # Turkish Draughts: 8 * 8 = 64 positions at depth 2
    assert board.perft(2) == 64


def test_board_hash():
    """Test Board is hashable."""
    board1 = kish.Board()
    board2 = kish.Board()
    assert hash(board1) == hash(board2)


def test_board_equality():
    """Test Board equality."""
    board1 = kish.Board()
    board2 = kish.Board()
    assert board1 == board2

    board3 = board1.apply(board1.actions()[0])
    assert board1 != board3
