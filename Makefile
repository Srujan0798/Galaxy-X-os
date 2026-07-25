# Galaxy-X-os Makefile

.PHONY: install split train evaluate gradcam app test lint validate clean

install:
	pip install -r requirements.txt

split:
	python3 src/prepare_data.py

train:
	python3 src/train.py

evaluate:
	python3 src/evaluate.py

gradcam:
	python3 src/gradcam.py

app:
	streamlit run app/app.py

test:
	python3 -m pytest tests/ -v

lint:
	ruff check src/ app/ tests/

validate:
	@echo "validate target disabled in CI — it downloads real data. Run locally: python3 src/prepare_data.py --per-class 10"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
