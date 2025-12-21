//! Benchmarks for the kish Turkish Draughts engine.
//!
//! Run with: `RUSTFLAGS="-C target-cpu=native" cargo bench`

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kish::{Board, Game, Square, Team};

/// Benchmark perft at various depths from initial position.
fn benchmark_perft(c: &mut Criterion) {
    let board = Board::new_default();
    let mut game = Game::new();

    let mut group = c.benchmark_group("Perft");

    for depth in [5, 6, 7] {
        group.bench_with_input(
            BenchmarkId::new("Board/depth", depth),
            &depth,
            |b, &depth| {
                b.iter(|| black_box(board.perft(black_box(depth))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("Game/depth", depth),
            &depth,
            |b, &depth| {
                b.iter(|| black_box(game.perft(black_box(depth))));
            },
        );
    }
    group.finish();
}

/// Benchmark perft from various positions (pawns, kings, captures).
fn benchmark_perft_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("Perft Scenarios");

    // Initial position (pawn-heavy)
    let initial = Board::new_default();
    group.bench_function("initial_d6", |b| {
        b.iter(|| black_box(initial.perft(6)));
    });

    // King endgame (4v4 kings)
    let king_endgame = Board::from_squares(
        Team::White,
        &[Square::A1, Square::A8, Square::D4, Square::E5],
        &[Square::H1, Square::H8, Square::D5, Square::E4],
        &[
            Square::A1,
            Square::A8,
            Square::D4,
            Square::E5,
            Square::H1,
            Square::H8,
            Square::D5,
            Square::E4,
        ],
    );
    group.bench_function("king_endgame_d6", |b| {
        b.iter(|| black_box(king_endgame.perft(6)));
    });

    // Mixed midgame (pawns + kings)
    let midgame = Board::from_squares(
        Team::White,
        &[
            Square::A2,
            Square::B2,
            Square::C3,
            Square::D4,
            Square::E4,
            Square::F3,
            Square::G2,
            Square::H2,
        ],
        &[
            Square::A7,
            Square::B7,
            Square::C6,
            Square::D5,
            Square::E5,
            Square::F6,
            Square::G7,
            Square::H7,
        ],
        &[Square::D4, Square::D5],
    );
    group.bench_function("mixed_midgame_d6", |b| {
        b.iter(|| black_box(midgame.perft(6)));
    });

    // Dense capture scenario
    let capture_heavy = Board::from_squares(
        Team::White,
        &[Square::A2, Square::C4, Square::E4, Square::G4],
        &[
            Square::B3,
            Square::C5,
            Square::D5,
            Square::E5,
            Square::F5,
            Square::G5,
            Square::H5,
        ],
        &[Square::C4, Square::E4],
    );
    group.bench_function("capture_heavy_d6", |b| {
        b.iter(|| black_box(capture_heavy.perft(6)));
    });

    // King flying capture chains
    let king_chains = Board::from_squares(
        Team::White,
        &[Square::A1],
        &[Square::A3, Square::C3, Square::C5, Square::E5, Square::E7],
        &[Square::A1],
    );
    group.bench_function("king_chains_d8", |b| {
        b.iter(|| black_box(king_chains.perft(8)));
    });

    group.finish();
}

/// Benchmark action generation for various board positions.
fn benchmark_action_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Action Generation");

    // Initial position
    let initial = Board::new_default();
    group.bench_function("initial", |b| {
        b.iter(|| black_box(initial.actions()));
    });

    // Mid-game position with captures
    let midgame = Board::from_squares(
        Team::White,
        &[Square::D4, Square::E4, Square::F4, Square::D5],
        &[Square::D6, Square::E6, Square::F6, Square::D7],
        &[Square::D4], // One white king
    );
    group.bench_function("midgame_with_captures", |b| {
        b.iter(|| black_box(midgame.actions()));
    });

    // Endgame with kings only
    let endgame = Board::from_squares(
        Team::White,
        &[Square::A1, Square::B1],
        &[Square::H8, Square::G8],
        &[Square::A1, Square::B1, Square::H8, Square::G8],
    );
    group.bench_function("endgame_kings", |b| {
        b.iter(|| black_box(endgame.actions()));
    });

    group.finish();
}

/// Benchmark board operations (apply, rotate).
fn benchmark_board_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("Board Operations");

    let board = Board::new_default();
    let actions = board.actions();
    let action = &actions[0];

    group.bench_function("apply", |b| {
        b.iter(|| black_box(board.apply(black_box(action))));
    });

    group.bench_function("rotate", |b| {
        b.iter(|| black_box(board.rotate()));
    });

    group.bench_function("status", |b| {
        b.iter(|| black_box(board.status()));
    });

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(100)
        .warm_up_time(Duration::from_secs(3))
        .measurement_time(Duration::from_secs(10));
    targets = benchmark_perft, benchmark_perft_scenarios, benchmark_action_generation, benchmark_board_ops
);

criterion_main!(benches);
