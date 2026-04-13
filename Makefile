.PHONY: install test run_all docs_serve clean

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

run_all:
	python main.py --all

docs_build:
	python scripts/build_docs_content.py
	mkdocs build

docs_serve: docs_build
	mkdocs serve

clean:
	rm -rf results/
	rm -rf site/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
