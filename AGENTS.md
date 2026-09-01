# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/isr/`. Trajectory resampling is implemented in `resample_trajectory.py`, with I/O, gripper, and kinematics helpers in neighboring modules. Installed command-line entry points are under `src/isr/cli/`; the top-level `cli/` directory contains standalone utility scripts. Tests mirror package responsibilities in `tests/test_*.py`. Use `data/example_1.json` for lightweight examples, and keep documentation images in `assets/`. Generated datasets and results belong in ignored `datasets/` and `outputs/` directories.

## Build, Test, and Development Commands

Create an editable development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

- `pytest` runs the complete test suite.
- `pytest tests/test_resample.py -q` runs one focused test module.
- `ruff check src tests cli` checks Python style and common errors.
- `isr-resample --help`, `isr-batch --help`, and `isr-visualize --help` smoke-test installed CLI entry points.
- `python -m build` creates distribution artifacts when `build` is installed.

Conda users can instead run `conda env create -f environment.yml && conda activate isr`; install the dev extras afterward when testing or linting.

## Coding Style & Naming Conventions

Target Python 3.9+ and use four-space indentation. Follow standard Python naming: `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Add type hints to public APIs and concise docstrings describing array shapes and return values. Keep CLI parsing in `src/isr/cli/` and reusable logic in package modules. Run Ruff before submitting changes.

## Testing Guidelines

Tests use pytest and NumPy assertions. Name files `test_<area>.py` and functions `test_<behavior>()`. Cover boundary inputs, invalid shapes, and numerical behavior when changing trajectory logic. No coverage threshold is configured, but every bug fix should include a regression test. Keep fixtures and sample arrays small and deterministic.

## Commit & Pull Request Guidelines

History uses short, imperative summaries such as `Remove pdfs folder`; no Conventional Commits scheme is established. Keep each commit focused and describe the observable change. Pull requests should explain motivation and approach, list verification commands, and link relevant issues. Include before/after images when visualization output changes, and call out any JSON schema or CLI compatibility impact.
