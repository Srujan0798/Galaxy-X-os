# Galaxy-X-os Makefile

.PHONY: install split train evaluate gradcam app test lint validate clean

install:
	pip install -r requirements.txt

split:
	python src/prepare_data.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py

gradcam:
	python src/gradcam.py

app:
	streamlit run app/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ app/ tests/

validate:
	@echo "validate target disabled in CI — it downloads real data. Run locally: python src/prepare_data.py --per-class 10"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
