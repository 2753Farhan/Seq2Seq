FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create directories
RUN mkdir -p outputs/models outputs/plots outputs/results data

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

CMD ["python", "src/main.py", "--train-all", "--evaluate", "--plot", "--save-results"]