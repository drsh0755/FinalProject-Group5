# Makefile for common tasks

.PHONY: help install test train app clean docker

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies"
	@echo "  make test         - Run tests"
	@echo "  make download     - Download data"
	@echo "  make preprocess   - Preprocess data"
	@echo "  make train        - Train model"
	@echo "  make app          - Run Streamlit app"
	@echo "  make docker       - Build Docker image"
	@echo "  make clean        - Clean generated files"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v --cov=. --cov-report=html

download:
	python scripts/download_data.py \
		--tickers AAPL MSFT GOOGL NVDA AMZN \
		--output data/raw/historical_data.csv

preprocess:
	python scripts/preprocess_data.py \
		--input data/raw/historical_data.csv \
		--output data/processed/features_with_sentiment.csv \
		--add-sentiment

train:
	python training/train_tft.py \
		--config training/train_config.yaml \
		--data data/processed/features_with_sentiment.csv

app:
	streamlit run app/main.py

docker:
	docker build -t stock-forecast-app -f app/Dockerfile .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
