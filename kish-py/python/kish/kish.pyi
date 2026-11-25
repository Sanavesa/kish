"""Type stubs for the kish Turkish Draughts engine."""

from enum import IntEnum
from typing import List, Optional

class Team(IntEnum):
    """Represents a player in the game (White or Black)."""

    White = 0
    """The white player (moves first)."""
    Black = 1
    """The black player."""

    def opponent(self) -> Team:
        """Returns the opponent team."""
        ...

class Square(IntEnum):
    """Represents a square on the 8x8 board (A1-H8)."""

    A1 = 0
    B1 = 1
    C1 = 2
    D1 = 3
    E1 = 4
    F1 = 5
    G1 = 6
    H1 = 7
    A2 = 8
    B2 = 9
    C2 = 10
    D2 = 11
    E2 = 12
    F2 = 13
    G2 = 14
    H2 = 15
    A3 = 16
    B3 = 17
    C3 = 18
    D3 = 19
    E3 = 20
    F3 = 21
    G3 = 22
    H3 = 23
    A4 = 24
    B4 = 25
    C4 = 26
    D4 = 27
    E4 = 28
    F4 = 29
    G4 = 30
    H4 = 31
    A5 = 32
    B5 = 33
    C5 = 34
    D5 = 35
    E5 = 36
    F5 = 37
    G5 = 38
    H5 = 39
    A6 = 40
    B6 = 41
    C6 = 42
    D6 = 43
    E6 = 44
    F6 = 45
    G6 = 46
    H6 = 47
    A7 = 48
    B7 = 49
    C7 = 50
    D7 = 51
    E7 = 52
    F7 = 53
    G7 = 54
    H7 = 55
    A8 = 56
    B8 = 57
    C8 = 58
    D8 = 59
    E8 = 60
    F8 = 61
    G8 = 62
    H8 = 63

    @staticmethod
    def from_notation(notation: str) -> Square:
        """Creates a square from algebraic notation (e.g., "d4", "H8").

        Raises:
            ValueError: If the notation is invalid.
        """
        ...

    @staticmethod
    def from_row_col(row: int, col: int) -> Square:
        """Creates a square from row and column indices (0-7 each).

        Raises:
            ValueError: If row or col is out of range.
        """
        ...

    def row(self) -> int:
        """Returns the row (rank) of this square (0-7, where 0 is rank 1)."""
        ...

    def col(self) -> int:
        """Returns the column (file) of this square (0-7, where 0 is file A)."""
        ...

    def notation(self) -> str:
        """Returns the algebraic notation for this square (e.g., "D4")."""
        ...

    def manhattan(self, other: Square) -> int:
        """Returns the Manhattan distance to another square.

        Useful for heuristics (e.g., distance to promotion row).
        """
        ...

    def to_mask(self) -> int:
        """Returns this square as a bitboard mask (u64 with one bit set)."""
        ...

    @staticmethod
    def from_mask(mask: int) -> Square:
        """Creates a square from a bitboard mask.

        The mask must have exactly one bit set.

        Raises:
            ValueError: If mask doesn't have exactly one bit set.
        """
        ...

class GameStatus:
    """Represents the current status of a game."""

    def is_in_progress(self) -> bool:
        """Returns true if the game is still in progress."""
        ...

    def is_draw(self) -> bool:
        """Returns true if the game ended in a draw."""
        ...

    def is_won(self) -> bool:
        """Returns true if the game has been won."""
        ...

    def is_over(self) -> bool:
        """Returns true if the game is over (draw or won)."""
        ...

    def winner(self) -> Optional[Team]:
        """Returns the winning team if the game was won, otherwise None."""
        ...

class Action:
    """Represents a move in the game.

    Use `notation()` to get the algebraic notation.
    Use `source()` and `destination()` to get the squares.
    """

    def source(self) -> Square:
        """Returns the source square of this move."""
        ...

    def destination(self) -> Square:
        """Returns the destination square of this move."""
        ...

    def is_capture(self) -> bool:
        """Returns true if this move is a capture."""
        ...

    def is_promotion(self) -> bool:
        """Returns true if this move results in promotion to king."""
        ...

    def capture_count(self) -> int:
        """Returns the number of pieces captured (0 for non-captures)."""
        ...

    def path(self) -> List[Square]:
        """Returns the full path of squares visited during this move."""
        ...

    def notation(self) -> str:
        """Returns the algebraic notation for this move (e.g., "d3-d4", "d4xd6")."""
        ...

    # =========================================================================
    # Bitboard access for ML
    # =========================================================================

    def captured_pieces(self) -> List[Square]:
        """Returns the captured pieces as a list of squares.

        Empty list for non-captures.
        """
        ...

    def captured_bitboard(self) -> int:
        """Returns the captured pieces as a bitboard (u64).

        Zero for non-captures.
        """
        ...

    def delta(self) -> tuple[int, int, int]:
        """Returns the action delta as a tuple of bitboards.

        Returns (white_delta, black_delta, kings_delta).
        XOR these with the board state to apply/unapply the action.
        """
        ...

    def delta_array(self) -> list[int]:
        """Returns the action delta as an array [white_delta, black_delta, kings_delta].

        Convenient for numpy: `np.array(action.delta_array(), dtype=np.uint64)`
        """
        ...

class Board:
    """The game board with piece positions and current turn.

    This is the main type for playing Turkish Draughts.
    Use `actions()` to get legal moves, `apply()` to make a move.
    """

    def __init__(self) -> None:
        """Creates a new board with the standard starting position."""
        ...

    @staticmethod
    def from_squares(
        turn: Team,
        white_squares: List[Square],
        black_squares: List[Square],
        king_squares: List[Square],
    ) -> Board:
        """Creates a board from custom piece positions.

        Args:
            turn: The team to move.
            white_squares: List of squares with white pieces.
            black_squares: List of squares with black pieces.
            king_squares: List of squares that are kings (must be subset of pieces).

        Raises:
            ValueError: If the position is invalid: a square holds both colours, a
                king is not on a piece, a side has more than 16 pieces, or a man is
                on its own promotion row.
        """
        ...

    @property
    def turn(self) -> Team:
        """Returns the current team's turn."""
        ...

    def actions(self) -> List[Action]:
        """Returns all legal actions from the current position."""
        ...

    def apply(self, action: Action) -> Board:
        """Applies an action and returns a new board with the turn swapped.

        Raises:
            ValueError: If the action is not legal in this position (e.g. it came
                from another position or was already applied).
        """
        ...

    def status(self) -> GameStatus:
        """Returns the current game status."""
        ...

    def white_pieces(self) -> List[Square]:
        """Returns the squares occupied by white pieces."""
        ...

    def black_pieces(self) -> List[Square]:
        """Returns the squares occupied by black pieces."""
        ...

    def kings(self) -> List[Square]:
        """Returns the squares occupied by kings."""
        ...

    def friendly_pieces(self) -> List[Square]:
        """Returns the squares occupied by the current player's pieces."""
        ...

    def hostile_pieces(self) -> List[Square]:
        """Returns the squares occupied by the opponent's pieces."""
        ...

    def rotate(self) -> Board:
        """Returns a rotated copy of the board (180 degrees)."""
        ...

    def perft(self, depth: int) -> int:
        """Runs a perft (performance test) at the given depth.

        Returns the number of leaf nodes (positions) at that depth.
        """
        ...

    # =========================================================================
    # Bitboard access for ML
    # =========================================================================

    def white_bitboard(self) -> int:
        """Returns the white pieces bitboard as a u64.

        Each bit represents a square (bit 0 = A1, bit 63 = H8).
        A set bit indicates a white piece on that square.
        """
        ...

    def black_bitboard(self) -> int:
        """Returns the black pieces bitboard as a u64."""
        ...

    def kings_bitboard(self) -> int:
        """Returns the kings bitboard as a u64.

        This is a subset of white_bitboard | black_bitboard.
        """
        ...

    def bitboards(self) -> tuple[int, int, int, int]:
        """Returns all bitboards as a tuple: (white, black, kings, turn).

        Turn is 0 for White, 1 for Black.
        Useful for ML feature extraction.
        """
        ...

    def to_array(self) -> list[int]:
        """Returns the board state as a flat array of 4 u64 values.

        Order: [white_pieces, black_pieces, kings, turn_as_u64]
        Convenient for numpy: `np.array(board.to_array(), dtype=np.uint64)`
        """
        ...

    @staticmethod
    def from_bitboards(turn: int, white: int, black: int, kings: int) -> Board:
        """Creates a board from raw bitboards.

        Args:
            turn: The team to move (0 = White, 1 = Black).
            white: White pieces bitboard.
            black: Black pieces bitboard.
            kings: Kings bitboard (must be subset of white | black).

        Raises:
            ValueError: If turn is not 0 or 1, or the position is invalid (see
                `from_squares`).
        """
        ...

class Game:
    """Full game with history tracking for proper draw detection.

    Use this when you need:
    - Threefold repetition detection
    - 50-move rule (insufficient progress)
    - Undo functionality

    For pure move generation (AI, perft), use `Board` directly.
    """

    def __init__(self) -> None:
        """Creates a new game with the standard starting position."""
        ...

    @staticmethod
    def from_board(board: Board) -> Game:
        """Creates a game from an existing board position."""
        ...

    def board(self) -> Board:
        """Returns the current board state."""
        ...

    @property
    def turn(self) -> Team:
        """Returns the current team's turn."""
        ...

    def actions(self) -> List[Action]:
        """Returns all legal actions from the current position."""
        ...

    def status(self) -> GameStatus:
        """Returns the current game status (including draw conditions)."""
        ...

    def make_move(self, action: Action) -> None:
        """Makes a move and updates the game state.

        Raises:
            ValueError: If the action is not legal in this position (e.g. it came
                from another position or was already played).
        """
        ...

    def undo_move(self) -> bool:
        """Undoes the last move. Returns true if a move was undone."""
        ...

    @property
    def halfmove_clock(self) -> int:
        """Returns the number of half-moves since the last capture."""
        ...

    @property
    def move_count(self) -> int:
        """Returns the number of moves made in this game."""
        ...

    def is_threefold_repetition(self) -> bool:
        """Returns true if the current position has occurred 3+ times."""
        ...

    def position_count(self) -> int:
        """Returns how many times the current position has occurred."""
        ...

    def clear_history(self) -> None:
        """Clears the game history and resets the halfmove clock."""
        ...

    def perft(self, depth: int) -> int:
        """Runs a perft (performance test) at the given depth."""
        ...
