.PHONY: up down migrate ingest test run lint clean download prepare-data train eval eval-cv

up:
	docker-compose up -d

down:
	docker-compose down

migrate:
	alembic upgrade head

ingest:
	python scripts/ingest_corpus.py

download:
	python scripts/download_public_data.py --corpus --wikipedia

download-corpus:
	python scripts/download_public_data.py --corpus --wikipedia

ingest-corpus:
	python scripts/ingest_corpus.py --rebuild

eval-rag:
	python evals/run_eval.py --stage rag

prepare-data:
	python scripts/prepare_deeppcb.py

train:
	python scripts/train_detector.py --run

eval:
	python evals/run_eval.py --stage synthetic

eval-cv:
	python evals/run_eval.py --stage cv --backend yolo --ultralytics-val

test:
	pytest tests/ -v --asyncio-mode=auto

run:
	uvicorn repair_agent.api.main:app --reload --port 8000

lint:
	ruff check src/ tests/ evals/ scripts/
	mypy src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache
