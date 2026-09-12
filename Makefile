.PHONY: install test run docker-build docker-run clean

install:
	pip install -r requirements.txt

test:
	pytest -v test_main.py

run:
	python main.py

docker-build:
	docker build -t water-quality-mlops .

docker-run:
	docker run --rm water-quality-mlops

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
