# Galaxy-X-os Makefile

.PHONY: install train evaluate gradcam app test lint clean validate

install:
	pip install -r requirements.txt

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
	ruff check src/ || true
	mypy src/ || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

validate:
	bash orchestrator/scripts/validate.sh
