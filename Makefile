install:
	poetry install --no-root --sync
	poetry run pre-commit install
	poetry run pre-commit autoupdate

docker-clean:
	docker stop resume-parser-backend-c || true && docker rm resume-parser-backend-c || true
	docker rmi resume-parser-backend || true

docker-image-cpu:
	make docker-clean
	docker image build -t resume-parser-backend:latest . --progress=plain

docker-container-cpu:
	docker container run -d --name resume-parser-backend-c -p 8082:8080 resume-parser-backend:latest

docker-image-gpu:
	make docker-clean
	docker image build -f gpu.Dockerfile -t resume-parser-backend:latest . --progress=plain

docker-container-gpu:
	docker container run -d --name resume-parser-backend-c -p 8082:8080 --runtime=nvidia --gpus all resume-parser-backend:latest

start:
	poetry run uvicorn main:app --reload --port 8082
