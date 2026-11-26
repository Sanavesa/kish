# kish

A high-performance Turkish Draughts (Dama) engine with Python bindings.

Built with Rust and PyO3 for maximum speed. Designed for both **UI applications** (simple API with algebraic notation) and **machine learning** (fast bitboard access).

## Installation

```bash
pip install kish
```

Or build from source:

```bash
cd kish-py
pip install maturin
maturin develop --release
```

## Quick Start

```python
import kish

# Create a new game with standard starting position
board = kish.Board()
print(f"Turn: {board.turn}")  # Turn: White

# Get all legal moves
actions = board.actions()
print(f"Legal moves: {len(actions)}")  # Legal moves: 23

# Make a move (returns new board - immutable)
new_board = board.apply(actions[0])
print(f"Move: {actions[0].notation()}")  # e.g., "a3-a4"

# Check game status
status = new_board.status()
if status.is_in_progress():
    print("Game continues")
elif status.is_draw():
    print("Draw!")
else:
    print(f"Winner: {status.winner()}")
```

## Game with History (Draw Detection)

Use `Game` for full game management with proper draw detection:

```python
import kish

game = kish.Game()

# Play moves
while not game.status().is_over():
    actions = game.actions()
    if not actions:
        break

    # Make move (mutates game state)
    game.make_move(actions[0])

    # Draw detection
    if game.is_threefold_repetition():
        print("Draw by repetition!")
        break

    if game.halfmove_clock >= 150:
        print("Draw: 150 plies without a capture!")
        break

# Undo moves
game.undo_move()
print(f"Moves played: {game.move_count}")
```

## Custom Positions

```python
import kish

# Create a custom position
board = kish.Board.from_squares(
    turn=kish.Team.White,
    white_squares=[kish.Square.D4, kish.Square.E3],
    black_squares=[kish.Square.D5, kish.Square.F6],
    king_squares=[kish.Square.D4],  # D4 is a king
)

# Query pieces
print(f"White pieces: {[str(s) for s in board.white_pieces()]}")
print(f"Kings: {[str(s) for s in board.kings()]}")
```

## Machine Learning

### Bitboard Access

Fast access to raw bitboard representation for neural network input:

```python
import kish
import numpy as np

board = kish.Board()

# Get individual bitboards (u64 integers)
white = board.white_bitboard()
black = board.black_bitboard()
kings = board.kings_bitboard()

# Get all at once (most efficient)
white, black, kings, turn = board.bitboards()

# As numpy array
arr = np.array(board.to_array(), dtype=np.uint64)
# arr = [white_pieces, black_pieces, kings, turn]

# Reconstruct from bitboards
board = kish.Board.from_bitboards(turn=0, white=white, black=black, kings=kings)
```

### Bit Plane Conversion for CNNs

```python
import numpy as np

def to_bit_planes(board):
    """Convert board to 4x8x8 tensor for CNN input."""
    w, b, k, turn = board.bitboards()
    planes = np.zeros((4, 8, 8), dtype=np.float32)

    for i in range(64):
        row, col = i // 8, i % 8
        planes[0, row, col] = (w >> i) & 1  # white pieces
        planes[1, row, col] = (b >> i) & 1  # black pieces
        planes[2, row, col] = (k >> i) & 1  # kings
    planes[3, :, :] = turn  # turn plane

    return planes

# Faster version using numpy
def to_bit_planes_fast(board):
    """Optimized bit plane conversion."""
    w, b, k, turn = board.bitboards()
    planes = np.zeros((4, 64), dtype=np.float32)
    planes[0] = np.unpackbits(np.array([w], '>u8').view(np.uint8))[::-1]
    planes[1] = np.unpackbits(np.array([b], '>u8').view(np.uint8))[::-1]
    planes[2] = np.unpackbits(np.array([k], '>u8').view(np.uint8))[::-1]
    planes[3] = turn
    return planes.reshape(4, 8, 8)
```

### Action Features for Policy Networks

```python
# Get action features for ML
for action in board.actions():
    # Source and destination
    src = action.source()
    dst = action.destination()

    # Move properties
    is_capture = action.is_capture()
    is_promotion = action.is_promotion()
    capture_count = action.capture_count()

    # Captured pieces (for reward shaping)
    captured = action.captured_pieces()      # List[Square]
    captured_bb = action.captured_bitboard() # u64

    # Raw delta for applying moves (XOR with board state)
    white_delta, black_delta, kings_delta = action.delta()
    delta_arr = np.array(action.delta_array(), dtype=np.uint64)
```

### Distance Heuristics

```python
# Manhattan distance for evaluation functions
sq1 = kish.Square.D4
sq2 = kish.Square.H8
distance = sq1.manhattan(sq2)  # 8

# Distance to promotion row
def distance_to_promotion(square, team):
    """Distance to back row for promotion."""
    if team == kish.Team.White:
        target_row = 7  # Row 8
    else:
        target_row = 0  # Row 1
    return abs(square.row() - target_row)
```

## Performance Testing

```python
import kish

board = kish.Board()

# Count positions at depth (perft)
nodes = board.perft(6)
print(f"Positions at depth 6: {nodes}")  # ~450 million nodes/sec
```

## Examples

The [`examples/`](https://github.com/sanavesa/kish/tree/master/kish-py/examples) directory contains runnable examples:

- `basic_game.py` - Simple game loop
- `custom_position.py` - Setting up custom board positions
- `ml_features.py` - Extracting features for machine learning
- `perft.py` - Performance testing with perft
- `random_playout.py` - Random game simulation

## API Reference

### Types

| Type | Description |
|------|-------------|
| `Team` | Enum: `White`, `Black` |
| `Square` | Enum: `A1` through `H8` (64 squares) |
| `GameStatus` | Game state with query methods |
| `Action` | Move with notation and bitboard access |
| `Board` | Immutable game board |
| `Game` | Mutable game with history tracking |

### Board Methods

| Method | Description |
|--------|-------------|
| `Board()` | Standard starting position |
| `Board.from_squares(...)` | Custom position from square lists |
| `Board.from_bitboards(...)` | Custom position from bitboards |
| `board.actions()` | Get legal moves |
| `board.apply(action)` | Make move (returns new board) |
| `board.status()` | Get game status |
| `board.perft(depth)` | Performance test |

### Board Bitboard Methods (ML)

| Method | Description |
|--------|-------------|
| `board.white_bitboard()` | White pieces as u64 |
| `board.black_bitboard()` | Black pieces as u64 |
| `board.kings_bitboard()` | Kings as u64 |
| `board.bitboards()` | Tuple: (white, black, kings, turn) |
| `board.to_array()` | Array: [white, black, kings, turn] |

### Action Methods

| Method | Description |
|--------|-------------|
| `action.source()` | Source square |
| `action.destination()` | Destination square |
| `action.notation()` | Algebraic notation (e.g., "d4xd6") |
| `action.is_capture()` | Is this a capture? |
| `action.is_promotion()` | Does piece promote? |
| `action.path()` | Full path of squares |

### Action Bitboard Methods (ML)

| Method | Description |
|--------|-------------|
| `action.captured_pieces()` | Captured squares as list |
| `action.captured_bitboard()` | Captured pieces as u64 |
| `action.delta()` | Tuple: (white_delta, black_delta, kings_delta) |
| `action.delta_array()` | Array: [white_delta, black_delta, kings_delta] |

### Game Methods

| Method | Description |
|--------|-------------|
| `Game()` | New game |
| `Game.from_board(board)` | From existing position |
| `game.make_move(action)` | Make move (mutates) |
| `game.undo_move()` | Undo last move |
| `game.is_threefold_repetition()` | Check repetition draw |
| `game.halfmove_clock` | Moves since last capture |
| `game.move_count` | Total moves made |

### Square Methods

| Method | Description |
|--------|-------------|
| `Square.from_notation("d4")` | Parse notation |
| `Square.from_row_col(row, col)` | From indices |
| `Square.from_mask(u64)` | From bitboard |
| `square.notation()` | To string (e.g., "D4") |
| `square.row()` | Row index (0-7) |
| `square.col()` | Column index (0-7) |
| `square.to_mask()` | To bitboard |
| `square.manhattan(other)` | Distance to other square |

## Bitboard Layout

```
Bit index = row * 8 + col

    A   B   C   D   E   F   G   H
  +---+---+---+---+---+---+---+---+
8 |56 |57 |58 |59 |60 |61 |62 |63 |  <- White promotes here
  +---+---+---+---+---+---+---+---+
7 |48 |49 |50 |51 |52 |53 |54 |55 |
  +---+---+---+---+---+---+---+---+
6 |40 |41 |42 |43 |44 |45 |46 |47 |
  +---+---+---+---+---+---+---+---+
5 |32 |33 |34 |35 |36 |37 |38 |39 |
  +---+---+---+---+---+---+---+---+
4 |24 |25 |26 |27 |28 |29 |30 |31 |
  +---+---+---+---+---+---+---+---+
3 |16 |17 |18 |19 |20 |21 |22 |23 |
  +---+---+---+---+---+---+---+---+
2 | 8 | 9 |10 |11 |12 |13 |14 |15 |
  +---+---+---+---+---+---+---+---+
1 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |  <- Black promotes here
  +---+---+---+---+---+---+---+---+
    A   B   C   D   E   F   G   H
```

## License

Apache-2.0
