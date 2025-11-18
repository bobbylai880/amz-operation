.PHONY: install-dev test

install-dev:
	python -m pip install -r requirements-dev.txt

test: install-dev
	pytest
