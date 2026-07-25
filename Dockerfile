# Galaxy-X-os Dockerfile
# NOTE: checkpoint + data volumes MUST be mounted at runtime:
#   - /app/checkpoints  (trained weights, e.g. best_model.pth)
#   - /app/data         (dataset: data/processed/{train,val,test}/<class>/*.png)
# Optional mount for sample images:
#   - /app/data/samples (for demo without full dataset)
#
# Build:  docker build -t galaxy-x-os .
# Run:    docker run -p 8501:8501 \
#           -v /path/to/checkpoints:/app/checkpoints \
#           -v /path/to/data:/app/data \
#           galaxy-x-os

FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app/ ./app/
COPY configs/ ./configs/

EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
