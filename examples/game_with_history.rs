//! Game with history example - proper draw detection and undo.
//!
//! Run with: `cargo run --example game_with_history`

use kish::{Game, GameStatus};

fn main() {
    println!("=== Game with History ===\n");

    let mut game = Game::new();

    println!("Playing moves with history tracking...\n");

    // Play some moves
    let mut move_count = 0;
    while game.status() == GameStatus::InProgress && move_count < 20 {
        let actions = game.actions();
        if actions.is_empty() {
            break;
        }

        // Pick the first move
        let action = &actions[0];
        let board = game.board();
        let detailed = action.to_detailed(board.turn, &board.state);

        println!(
            "Move {}: {} plays {}",
            move_count + 1,
            board.turn,
            detailed.to_notation()
        );

        game.make_move(action);
        move_count += 1;

        // Check for threefold repetition
        if game.is_threefold_repetition() {
            println!("\nDraw by threefold repetition!");
            break;
        }

        // Check for 150-move rule
        if game.halfmove_clock() >= 150 {
            println!("\nDraw by 150-move rule!");
            break;
        }
    }

    println!();
    println!("Game state after {} moves:", game.move_count());
    println!("  Halfmove clock: {}", game.halfmove_clock());
    println!(
        "  Position occurred {} time(s)",
        game.position_occurrence_count()
    );
    println!("  Status: {}", game.status());

    // Demonstrate undo
    println!();
    println!("Undoing last 3 moves...");
    for _ in 0..3 {
        if game.undo_move() {
            println!("  Undone. Moves remaining: {}", game.move_count());
        } else {
            println!("  No more moves to undo.");
            break;
        }
    }

    println!();
    println!("Final position:");
    println!("{}", game.board());
}
