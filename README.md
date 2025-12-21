# Kish

[![Crates.io](https://img.shields.io/crates/v/kish.svg)](https://crates.io/crates/kish)
[![Documentation](https://docs.rs/kish/badge.svg)](https://docs.rs/kish)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A high-performance Turkish Draughts (Dama) engine written in Rust.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Rules Summary](#rules-summary)
- [Python Bindings](#python-bindings)
- [Contributing](#contributing)
- [License](#license)

## Overview

Turkish Draughts (Dama) is a variant of checkers with **orthogonal movement** (horizontal and vertical) rather than diagonal. This engine implements all official rules and is optimized for speed using bitboard representation.

```
   A B C D E F G H
8  . . . . . . . .  8   <- Black promotes here
7  b b b b b b b b  7
6  b b b b b b b b  6
5  . . . . . . . .  5
4  . . . . . . . .  4
3  w w w w w w w w  3
2  w w w w w w w w  2
1  . . . . . . . .  1   <- White promotes here
   A B C D E F G H
```

## Features

- **Complete Rules** - Movement, captures, promotion, all draw conditions
- **Bitboard Engine** - 64-bit representation for efficient computation
- **~330M nodes/sec** - Optimized move generation and perft
- **Draw Detection** - Threefold repetition and 50-move rule
- **Algebraic Notation** - Standard move notation support

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
kish = "1.0"
```

Or via cargo:

```bash
cargo add kish
```

## Quick Start

```rust
use kish::{Game, GameStatus};

fn main() {
    let mut game = Game::new();

    // Game loop
    while !game.status().is_over() {
        let actions = game.actions();

        // Pick a move (here: first legal move)
        if let Some(action) = actions.first() {
            game.make_move(action);
        }
    }

    match game.status() {
        GameStatus::Won(team) => println!("{} wins!", team),
        GameStatus::Draw => println!("Draw!"),
        _ => {}
    }
}
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| [`Board`](https://docs.rs/kish/latest/kish/struct.Board.html) | Lightweight board state for AI/search |
| [`Game`](https://docs.rs/kish/latest/kish/struct.Game.html) | Full game with history and draw detection |
| [`Action`](https://docs.rs/kish/latest/kish/struct.Action.html) | Compact move (XOR delta representation) |
| [`ActionPath`](https://docs.rs/kish/latest/kish/struct.ActionPath.html) | Move with full path for notation |
| [`Square`](https://docs.rs/kish/latest/kish/enum.Square.html) | Board square (A1-H8) |
| [`Team`](https://docs.rs/kish/latest/kish/enum.Team.html) | White or Black |
| [`GameStatus`](https://docs.rs/kish/latest/kish/enum.GameStatus.html) | InProgress, Draw, or Won |

### Board vs Game

| Use Case | Type |
|----------|------|
| AI search (alpha-beta, MCTS) | `Board` |
| Perft testing | `Board` |
| High-speed simulations | `Board` |
| Full games with draw detection | `Game` |
| Undo/redo support | `Game` |

### Move Notation

```rust
use kish::{ActionPath, Square};

// Simple move: e3-e4
let mv = ActionPath::new_move(Square::E3, Square::E4, false);

// Capture: d4xd6
let cap = ActionPath::new_capture(Square::D4, &[Square::D6], false);

// Multi-capture: b3xd3xd5
let multi = ActionPath::new_capture(Square::B3, &[Square::D3, Square::D5], false);

// Promotion: c7-c8=K
let promo = ActionPath::new_move(Square::C7, Square::C8, true);

println!("{}", promo.to_notation()); // "c7-c8=K"
```

## Performance

Benchmarks run on AMD Ryzen 9 5900X with `RUSTFLAGS="-C target-cpu=native"`. See [PERFORMANCE.md](PERFORMANCE.md) for the full optimization history.

### Perft (Move Generation)

| Depth | Nodes | Time | Nodes/sec |
|-------|------:|-----:|----------:|
| 5 | 85,090 | 228 µs | 373M |
| 6 | 931,312 | 2.7 ms | 341M |
| 7 | 10,782,382 | 32.5 ms | 332M |

### Scenario Benchmarks

| Position | Depth | Time |
|----------|------:|-----:|
| Initial (pawns) | 6 | 2.7 ms |
| King endgame (4v4) | 6 | 12.7 ms |
| Mixed midgame | 6 | 2.5 ms |
| Capture-heavy | 6 | 103 µs |
| King chains | 8 | 13.6 ms |

### Micro-operations

| Operation | Time |
|-----------|-----:|
| Generate moves (initial) | 37 ns |
| Generate moves (midgame) | 31 ns |
| Apply action | 2.2 ns |
| Board rotation | 2.0 ns |
| Check status | 37 ns |

Run benchmarks yourself:

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Rules Summary

See [RULES.md](RULES.md) for the complete official rules.

### Movement
- **Men**: Forward and sideways (not backward), one square
- **Kings**: All four orthogonal directions, any distance (flying kings)

### Capturing
- Captures are **mandatory** when available
- Must choose the path that captures the **maximum pieces**
- No 180° turns during multi-capture sequences
- Captured pieces are removed **immediately**

### Promotion
- Men promote to kings on the opponent's back row
- During multi-capture: continues as man, promotes at end

### Draw Conditions
- One piece each (automatic draw)
- Threefold repetition (same position 3 times)
- 50 plies without capture (insufficient progress)

## Python Bindings

Python bindings are available via PyPI:

```bash
pip install kish
```

See [kish-py/README.md](kish-py/README.md) for Python documentation and ML examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and release instructions.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.
