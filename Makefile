# Galaxy-X-os Makefile

.PHONY: install split train train-head evaluate gradcam app test lint clean

install:
	pip install -r requirements.txt

split:
	python src/generate_splits.py

train:
	python src/train.py

train-head:
	python src/train_head.py

evaluate:
	python src/evaluate.py

gradcam:
	python src/gradcam.py

app:
	streamlit run app/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ || true
	mypy src/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
