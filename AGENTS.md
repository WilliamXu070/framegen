# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the primary entry point for training, inference, testing, and demos.
- `src/` contains application code: training (`trainer.py`, `trainer_ldmvfi.py`), inference (`inference.py`, `inference_ldmvfi.py`), models (`src/models/`), and data utilities (`src/data/`).
- `configs/` stores YAML configuration files (start with `configs/default.yaml`).
- `data/`, `models/`, `logs/`, `output/`, and `demo*/` hold datasets, checkpoints, logs, and generated media.
- Root-level `test_*.py` files are runnable test scripts.

## Build, Test, and Development Commands
- `python main.py --mode train --config configs/default.yaml`: train a model.
- `python main.py --mode inference --input input.mp4 --output output.mp4 --model models/best_model.pth`: run inference.
- `python main.py --mode test --test-video sample_video.mp4 --test-output-dir demo --test-fps-multiplier 2`: generate demo outputs.
- `python test_light_loading.py`: run a standalone test script (repeat for other `test_*.py`).

## Coding Style & Naming Conventions
- Python uses 4-space indentation; keep lines readable and avoid deeply nested blocks.
- Use `snake_case` for functions/variables and `PascalCase` for classes.
- Optional tooling in `setup.py`: `black`, `flake8`, and `mypy` (install via extras if desired).

## Testing Guidelines
- Tests are plain Python scripts named `test_*.py`; they should be runnable with `python <script>`.
- There is no enforced coverage threshold; add focused checks for data loading, model I/O, and inference outputs.

## Commit & Pull Request Guidelines
- Git history shows no strict commit convention; keep messages short, descriptive, and in present tense.
- PRs should explain the change, note any new configs or data requirements, and include example outputs for visual changes (e.g., demo frames or videos).

## Configuration & Data Notes
- Prefer editing YAML in `configs/` over hardcoding parameters.
- Large datasets and generated artifacts belong under `data/`, `output/`, or `demo*/` and should stay out of commits.
