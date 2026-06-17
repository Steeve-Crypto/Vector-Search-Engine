# Contributing to Vector Search Engine

Thank you for your interest in contributing!

## Development Setup

1. Install Rust (stable, 1.80+)
2. Clone the repo
3. `cargo build`
4. Download model: `cargo run -- download-model` (or place manually)
5. Run tests: `cargo test`
6. Run benchmarks: `cargo bench`

## Code Style

- Follow Rust standard style (`cargo fmt`, `cargo clippy`)
- Use `thiserror` for errors
- Add tests for new features
- Keep dependencies minimal
- Document public APIs

## Pull Requests

- Reference issues
- Include benchmarks/eval for perf changes
- Update README/plan.md if phases/docs change
- Small, focused PRs preferred

## Reporting Issues

- Use GitHub issues
- Include repro steps, `cargo version`, OS

## Architecture Decisions

See `docs/adr/` for ADRs.

## License

MIT - see LICENSE
