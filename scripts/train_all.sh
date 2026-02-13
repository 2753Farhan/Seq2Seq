#!/bin/bash

echo "==================================="
echo "Seq2Seq Code Generation Assignment"
echo "==================================="
echo ""

mkdir -p outputs/models outputs/plots outputs/results data

if command -v docker &> /dev/null; then
    echo "Docker detected. Running in Docker container..."
    echo ""
    
    if docker run --gpus all nvidia/cuda:11.0-base nvidia-smi &> /dev/null; then
        echo "GPU support enabled"
        docker-compose -f docker/docker-compose.yml up --build
    else
        echo "Warning: GPU support not available. Running on CPU (will be slow)"
        docker-compose -f docker/docker-compose.yml run seq2seq-codegen
    fi
else
    echo "Docker not detected. Running locally..."
    echo ""
    
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    source venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    echo "Downloading dataset..."
    python scripts/download_data.py
    
    echo ""
    echo "Training all models..."
    python src/main.py --train-all --evaluate --plot --save-results
    
    echo ""
    echo "Training complete! Check outputs/ directory for results."
fi