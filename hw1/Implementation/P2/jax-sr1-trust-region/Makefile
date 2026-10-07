.PHONY: install test bench clean

install:
	python -m pip install -e '.[test]'

test:
	pytest -q

bench:
	python -m trsr1 --platform cpu --precision 64

clean:
	rm -rf .pytest_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
