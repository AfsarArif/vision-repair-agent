.PHONY: up down migrate ingest test run lint clean

up:
	docker-compose up -d

down:
	docker-compose down

migrate:
	alembic upgrade head

ingest:
	python scripts/ingest_corpus.py

test:
	pytest tests/ -v --asyncio-mode=auto

run:
	uvicorn repair_agent.api.main:app --reload --port 8000

lint:
	ruff check src/ tests/
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
