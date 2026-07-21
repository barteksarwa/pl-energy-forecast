.PHONY: setup test smoke dry-run backtest lint viz backfill

viz:
	uv run python -m src.viz.make_all

viz-diag:
	uv run python -m src.viz.diagnostics

backfill:
	uv run python -m src.ingestion.backfill

setup:
	uv sync

test:
	uv run pytest -q -m "not smoke"

smoke:
	uv run pytest -q -m smoke

dry-run:
	uv run python -m src.pipeline.daily_run

backtest:
	uv run python -m src.evaluation.run_backtest

lint:
	uv run ruff check src tests
