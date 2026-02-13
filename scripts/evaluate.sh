#!/bin/bash

echo "Evaluating trained models..."
python src/main.py --evaluate --plot --save-results
echo "Evaluation complete!"