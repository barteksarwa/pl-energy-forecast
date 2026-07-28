.PHONY: setup test smoke dry-run backtest lint viz backfill check-docs

viz:
	uv run python -m src.viz.make_all

viz-diag:
	uv run python -m src.viz.diagnostics

backfill:
	uv run python -m src.ingestion.backfill

setup:
	uv sync

# Deep tests run in a separate process: torch + LightGBM in one process
# double-load OpenMP on macOS and segfault.
test:
	uv run pytest -q -m "not smoke" --ignore=tests/test_deep_data.py
	uv run pytest -q -m "not smoke" tests/test_deep_data.py

smoke:
	uv run pytest -q -m smoke

dry-run:
	uv run python -m src.pipeline.daily_run

backtest:
	uv run python -m src.evaluation.run_backtest

lint:
	uv run ruff check src tests scripts

# Every headline number in RESULTS/BENCHMARK/README, re-read from the
# artifact CSVs it was copied from. Fails on any drift. (VALIDATION.md
# recommendation 5.)
check-docs:
	uv run python scripts/check_doc_numbers.py
