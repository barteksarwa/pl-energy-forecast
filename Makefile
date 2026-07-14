.PHONY: setup test smoke dry-run backtest lint viz backfill

viz:
	uv run python -m src.viz.make_all

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
	@echo "Backtest lands in Milestone 3."

lint:
	uv run ruff check src tests
