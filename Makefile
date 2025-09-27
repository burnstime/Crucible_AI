up:
	docker compose up --build -d

test:
	pytest -q

lint:
	black --check crucible && flake8 crucible

build:
	docker build -t crucible_ai .
