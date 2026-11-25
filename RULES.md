# Turkish Checkers (Dama) - Official Rules

## Sources

This document compiles rules from the following authoritative sources:

1. [Wikipedia - Turkish Draughts](https://en.wikipedia.org/wiki/Turkish_draughts)
2. [FMJD (World Draughts Federation) - Turkish Draughts](https://www.fmjd.org/?p=turkish)
3. [Ludoteka - Turkish Draughts Rules](https://www.ludoteka.com/games/turkish-draughts/rules)
4. [Gambiter - Turkish Draughts](https://gambiter.com/checkers/Turkish_draughts.html)
5. [MindSports - Dama (Turkish Draughts)](https://mindsports.nl/index.php/on-the-evolution-of-draughts-variants/draughts-variants/502-dama_t)
6. [Checkers & Draughts Wiki - Dama](https://checkers.fandom.com/wiki/Dama)
7. [PlayOK - Turkish Dama](https://www.playok.com/en/turkishdama/)
8. [Bonaludo - Orthogonal Draughts](https://bonaludo.wordpress.com/2016/03/31/orthogonal-draughts-on-8x8-board-turkish-draughts-croda-and-dameo/)
9. [Draughts.github.io - Turkish Draughts](https://draughts.github.io/turkish-draughts.html)
10. [Grokipedia - Turkish Draughts](https://grokipedia.com/page/Turkish_draughts)
11. [Turkish Dama Rules (Bektash Güldeste)](https://www.scribd.com/document/356142946/Turkish-Dama-dama-Oklu)

---

## 1. Overview

Turkish Draughts, known as **Dama** (or **Türk Daması**) in Turkey, is a variant of checkers/draughts popular throughout Turkey, Greece, Egypt, and the broader Middle East region. The game is distinguished from standard checkers by its **orthogonal movement** (horizontal and vertical) rather than diagonal movement.

Since 2014, official World Championships have been held annually, with the first championship taking place in İzmir, Turkey.

---

## 2. Terminology

- **Ply** (also **Half-move**): A single action by one player. When White moves a piece, that's one ply. When Black responds, that's another ply. This term is unambiguous and preferred in game tree analysis and computer implementations.
- **Move** (also **Full move**): A pair of plies — one action by White followed by one action by Black. In game notation, moves are numbered sequentially (e.g., "1. d3-d4 d6-d5" is move 1, consisting of two plies). Note: In casual speech, "move" is often used loosely to mean a single ply.
- **Turn**: A player's opportunity to act. Equivalent to one ply. Players alternate turns, with White taking the first turn.
- **Man** (plural: **Men**): An unpromoted piece, also called a "pawn" or "regular piece."
- **King** (also **Dama**): A promoted piece with enhanced movement abilities. Called "Dama" (meaning "Lady") in Turkish.
- **Capture**: Jumping over and removing an opponent's piece from the board.
- **Multi-capture** (also **Chain capture**): A sequence of consecutive captures made in a single turn.
- **Flying capture**: A king's ability to capture from any distance and land on any empty square beyond the captured piece.
- **Promotion**: The transformation of a man into a king upon reaching the opponent's back row.
- **Promotion row** (also **Back row**, **King row**): The opponent's starting row where men are promoted (row 8 for White, row 1 for Black).
- **Orthogonal**: Movement along horizontal (ranks) or vertical (files) lines, as opposed to diagonal movement.
- **Stalemate** (also **Block**): A situation where a player has pieces but cannot make any legal move, resulting in a loss for that player.

---

## 3. Equipment

### 3.1 The Board
- Standard **8×8 board** with 64 squares
- Traditionally, the board is **mono-colored** (single color), though checkered boards may also be used
- When using a checkered board, the bottom-left square should be dark

### 3.2 The Pieces
- Each player has **16 pieces** (also called "men")
- One player plays **White**, the other plays **Black**
- Promoted pieces are called **Kings** (or **Dama**, meaning "Lady")

---

## 4. Initial Setup

```
    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 |   |   |   |   |   |   |   |   |  ← Black's back row (promotion row for White)
  +---+---+---+---+---+---+---+---+
7 | b | b | b | b | b | b | b | b |  ← Black pieces (row 7)
  +---+---+---+---+---+---+---+---+
6 | b | b | b | b | b | b | b | b |  ← Black pieces (row 6)
  +---+---+---+---+---+---+---+---+
5 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
4 |   |   |   |   |   |   |   |   |
  +---+---+---+---+---+---+---+---+
3 | w | w | w | w | w | w | w | w |  ← White pieces (row 3)
  +---+---+---+---+---+---+---+---+
2 | w | w | w | w | w | w | w | w |  ← White pieces (row 2)
  +---+---+---+---+---+---+---+---+
1 |   |   |   |   |   |   |   |   |  ← White's back row (promotion row for Black)
  +---+---+---+---+---+---+---+---+
    a   b   c   d   e   f   g   h
```

- **White pieces**: Placed on all squares of rows 2 and 3 (16 pieces total)
- **Black pieces**: Placed on all squares of rows 6 and 7 (16 pieces total)
- **Back rows (1 and 8)**: Initially empty - these are the promotion rows
- **White moves first**

---

## 5. Movement Rules

### 5.1 Movement Direction
All movement in Turkish Draughts is **orthogonal** (along ranks and files), **never diagonal**. This is a key distinction from most other draughts variants.

### 5.2 Men (Unpromoted Pieces)

**Non-capturing moves:**
- A man may move **one square** in any of the following directions:
  - Forward (toward the opponent's side)
  - Left (sideways)
  - Right (sideways)
- A man **cannot move backward**

**Visual representation of man's movement:**
```
        ↑
        |
    ← ─ M ─ →

(M = Man, arrows show valid move directions)
```

### 5.3 Kings (Promoted Pieces)

**Non-capturing moves:**
- A king may move **any number of unoccupied squares** in any orthogonal direction:
  - Forward
  - Backward
  - Left
  - Right
- The king moves like a **rook in chess** (but only along ranks and files)

**Visual representation of king's movement:**
```
        ↑
        |
    ← ─ K ─ →
        |
        ↓

(K = King, arrows show valid move directions, extends to any distance)
```

---

## 6. Capturing Rules

### 6.1 Mandatory Capture
- **Captures are mandatory**: If a capture is available, it must be taken
- A player cannot make a non-capturing move if a capture is possible

### 6.2 Men's Captures

- A man captures by **jumping over** an adjacent opponent's piece to an empty square immediately beyond it
- Men can only capture in three directions:
  - Forward
  - Left (sideways)
  - Right (sideways)
- **Men cannot capture backward**
- **Men cannot capture diagonally**

**Visual representation of man's capture directions:**
```
        ↑
        |
    ← ─ M ─ →

(M = Man, arrows show valid capture directions)
```

### 6.3 King's Captures

- A king captures by jumping over an opponent's piece along a rank or file
- The opponent's piece can be **any number of squares away** (as long as all squares between the king and the target are empty)
- After capturing, the king may land on **any empty square** beyond the captured piece along the same line
- Kings can capture in **all four orthogonal directions** (forward, backward, left, right)

### 6.4 Immediate Removal Rule

**Captured pieces are removed from the board immediately after being jumped**, before the capturing piece continues its sequence. This is a distinctive feature of Turkish Draughts.

**Consequence**: Because pieces are removed immediately:
- It is possible to **cross a square previously occupied by a captured piece** during the same multi-capture sequence
- This can open up additional jumps that were previously impossible

### 6.5 Multi-Capture (Chain Captures)

- If after making a capture, the capturing piece can make another capture, it **must continue capturing**
- The entire sequence of captures counts as **one move**
- The capturing piece continues jumping until no more captures are available

### 6.6 Maximum Capture Rule (Majority Rule)

- If multiple capture sequences are possible, the player **must choose the sequence that captures the most pieces**
- **No distinction is made between men and kings** when counting pieces - each counts as one piece
- If multiple sequences capture the same maximum number of pieces, the player may **freely choose** which sequence to take

### 6.7 180-Degree Turn Prohibition

**Within a multi-capture sequence, turning 180 degrees between two consecutive captures is NOT allowed.**

This means:
- If you jump forward, your next jump in the sequence cannot be directly backward
- If you jump left, your next jump cannot be directly right (and vice versa)
- You may turn 90 degrees between jumps (e.g., forward then left)

---

## 7. Promotion

### 7.1 Basic Promotion
- When a man reaches the **opponent's back row** (row 8 for White, row 1 for Black), it is **promoted to a king**
- The king gains enhanced movement and capture abilities as described above

### 7.2 Promotion During Capture Sequences

- If a man reaches the back row **during a capture sequence**, it must **complete the sequence as a man** (using only forward and sideways adjacent jumps)
- The promotion to king takes effect **at the end of the turn**
- The entire sequence still counts as one move

---

## 8. Winning the Game

A player **wins** the game by:

1. **Capturing all opponent's pieces** - The opponent has no pieces remaining on the board

2. **Blocking all opponent's pieces** - The opponent has pieces remaining but none of them can make a legal move when it is their turn (all are blocked)

---

## 9. Draw Conditions

The game is a **draw** under the following conditions:

### 9.1 Mutual Agreement
Both players agree to a draw.

### 9.2 Threefold Repetition
The same position occurs **three times** with the same player to move. The positions do not need to be consecutive.

### 9.3 One Piece Each
Both players have only **one piece remaining** each (regardless of whether they are men or kings) — unless the player to move can capture the opponent's last piece immediately, which wins (Section 8.1).

### 9.4 Insufficient Progress (Optional/Tournament Rule)
A draw is declared after **150 consecutive plies** (75 full moves) without any capture (only captures reset the count). This prevents indefinitely prolonged endgames.

### 9.5 Mutual Block
Being blocked is only decided on a player's own turn: the player to move who has no legal move loses (Section 8.2), even if the opponent would also be blocked. A player whose opponent is blocked must still make a move, which may free the opponent.

---

## 10. Notation

Moves in Turkish Draughts can be recorded using **algebraic notation**, similar to chess notation.

### 10.1 Square Identification

Each square is identified by a coordinate consisting of:
- **File (column)**: Letters `a` through `h` (left to right from White's perspective)
- **Rank (row)**: Numbers `1` through `8` (bottom to top from White's perspective)

Example: `d4` refers to the square in column d, row 4.

```
    a   b   c   d   e   f   g   h
  +---+---+---+---+---+---+---+---+
8 |a8 |b8 |c8 |d8 |e8 |f8 |g8 |h8 |
  +---+---+---+---+---+---+---+---+
7 |a7 |b7 |c7 |d7 |e7 |f7 |g7 |h7 |
  +---+---+---+---+---+---+---+---+
  ...
  +---+---+---+---+---+---+---+---+
1 |a1 |b1 |c1 |d1 |e1 |f1 |g1 |h1 |
  +---+---+---+---+---+---+---+---+
```

### 10.2 Move Notation

| Move Type | Format | Example | Description |
|-----------|--------|---------|-------------|
| Non-capturing move | `from-to` | `d3-d4` | Man moves from d3 to d4 |
| Single capture | `fromxto` | `d4xd6` | Piece on d4 captures piece on d5, lands on d6 |
| Multi-capture | `fromxmidxto` | `d4xd6xf6` | Piece captures two pieces in sequence |
| Promotion | `from-to=K` | `d7-d8=K` | Man moves to d8 and becomes a king |
| Capture with promotion | `fromxto=K` | `c7xc8=K` | Man captures and promotes |

### 10.3 Notation Examples

**Example 1: Simple move**
- `e3-e4` — White man moves forward from e3 to e4

**Example 2: Sideways move**
- `d4-e4` — Man moves right from d4 to e4

**Example 3: Single capture**
- `d4xd6` — Piece on d4 jumps over enemy on d5, landing on d6

**Example 4: Multi-capture sequence**
- `b3xd3xd5` — Piece captures two enemies: first moving right (b3 to d3), then forward (d3 to d5)

**Example 5: King long-range move**
- `a1-a7` — King moves from a1 to a7 (six squares forward)

**Example 6: Promotion**
- `c7-c8=K` — White man reaches c8 and is promoted to king

### 10.4 Game Record Format

A complete game can be recorded by numbering each pair of moves:

```
1. d3-d4   d6-d5
2. c3-c4   e6-e5
3. c4xc6   d7xd5
...
```

Each line shows the move number, White's move, then Black's move.

---

## 11. Quick Reference Card

### Valid Moves for Men
- Move: Forward, Left, Right (1 square)
- Capture: Forward, Left, Right (jump over adjacent piece)
- Cannot: Move or capture backward or diagonally

### Valid Moves for Kings
- Move: Forward, Backward, Left, Right (any number of squares)
- Capture: Forward, Backward, Left, Right (long-range jump)
- Can land on any empty square beyond captured piece

### Must Remember
1. Captures are mandatory
2. Must capture maximum pieces when multiple options exist
3. Captured pieces removed immediately
4. No 180° turns during multi-captures
5. One piece each = Draw

---
