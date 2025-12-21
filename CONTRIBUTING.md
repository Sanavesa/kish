# Contributing to Kish

## Development Setup

```bash
git clone https://github.com/sanavesa/kish.git
cd kish

# Install pre-commit hooks
pip install pre-commit
pre-commit install
pre-commit install --hook-type pre-push

# Run tests
cargo test

# Run benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench

# Build Python bindings (for development)
cd kish-py
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest tests/ -v
```

## Pre-commit Hooks

The project uses [pre-commit](https://pre-commit.com/) to run checks automatically:

**On commit:**
- `cargo fmt` - Format Rust code
- `cargo clippy` - Lint Rust code
- `ruff` - Format and lint Python code
- Trailing whitespace, YAML/TOML validation, etc.

**On push:**
- `cargo test` - Run Rust tests

To run all hooks manually:

```bash
pre-commit run --all-files
```

## Project Structure

```
kish/
├── src/                  # Rust library
├── examples/             # Rust examples
├── benches/              # Benchmarks
├── kish-py/              # Python bindings
│   ├── src/              # PyO3 bindings
│   ├── python/kish/      # Python package (stubs)
│   ├── tests/            # Python tests
│   └── examples/         # Python examples
└── scripts/              # Release automation
```

## Releasing

Releases are automated via GitHub Actions. A single release publishes to both crates.io and PyPI.

### Setup (one-time)

1. **crates.io**: Create a token at https://crates.io/settings/tokens and add it as `CARGO_REGISTRY_TOKEN` in GitHub repo secrets
2. **PyPI**: Configure trusted publishing at https://pypi.org/manage/account/publishing/ with:
   - Project: `kish`
   - Owner: your GitHub username
   - Repository: `kish`
   - Workflow: `release.yml`

### Release Process

```bash
# Ensure all changes are committed
git status  # should be clean

# Bump version and create tag
./scripts/release.sh patch   # or: minor, major, 1.2.3

# Push to trigger CI/CD
git push origin master --tags
```

The release script will:
1. Update version in `Cargo.toml`, `kish-py/Cargo.toml`, and `kish-py/pyproject.toml`
2. Create a commit with message "Release vX.Y.Z"
3. Create a git tag `vX.Y.Z`

The GitHub Actions workflow will then:
1. Publish the Rust crate to **crates.io**
2. Build Python wheels for Linux (x86_64, aarch64), macOS (x86_64, arm64), and Windows
3. Publish Python wheels to **PyPI**

### Dry Run

Preview what the release script will do without making changes:

```bash
./scripts/release.sh patch --dry-run
```
