//! Perft example - performance testing for move generation.
//!
//! Run with: `cargo run --release --example perft`

use kish::Board;
use std::time::Instant;

/// Format a number with comma separators (e.g., 1234567 -> "1,234,567").
fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result
}

/// Known perft values for the Turkish Draughts standard starting position.
const PERFT_VALUES: &[(u64, u64)] = &[
    (0, 1),
    (1, 8),
    (2, 64),
    (3, 708),
    (4, 7_538),
    (5, 85_090),
    (6, 931_312),
    (7, 10_782_308),
    (8, 123_285_642),
    (9, 1_454_012_156),
    (10, 16_989_029_970),
    (11, 204_346_895_344),
    (12, 2_454_694_682_456),
];

fn main() {
    println!("=== Perft (Performance Test) ===\n");
    println!("Counting positions at each depth from standard starting position.\n");

    let board = Board::new_default();

    println!(
        "{:<8} {:<18} {:<12} {:<18} Correct",
        "Depth", "Nodes", "Time (s)", "Nodes/sec"
    );
    println!("{}", "-".repeat(70));

    for &(depth, expected) in PERFT_VALUES {
        let start = Instant::now();
        let nodes = board.perft(depth);
        let elapsed = start.elapsed().as_secs_f64();

        let nps = if elapsed > 0.0 {
            nodes as f64 / elapsed
        } else {
            f64::INFINITY
        };

        let correct = if nodes == expected { "Yes" } else { "NO!" };

        println!(
            "{:<8} {:<18} {:<12.3} {:<18} {}",
            depth,
            format_with_commas(nodes),
            elapsed,
            format_with_commas(nps as u64),
            correct
        );

        // Stop if taking too long
        // if elapsed > 120.0 {
        //     println!("\nStopping at depth {depth} (>120s)");
        //     break;
        // }
    }

    println!();
    println!("Tip: Run with --release for optimized performance.");
}
