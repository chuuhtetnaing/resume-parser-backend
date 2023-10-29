install:
	poetry install --no-root --sync
	poetry run pre-commit install
	poetry run pre-commit autoupdate

start:
	poetry run uvicorn main:app --reload --port 8082
