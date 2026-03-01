# **📘 README.md - Complete Guide**

```markdown
# Seq2Seq Text-to-Python Code Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 **Assignment Overview**

This project implements and compares three Sequence-to-Sequence (Seq2Seq) models for generating Python code from natural language docstrings:

| Model | Description | Key Feature |
|-------|-------------|-------------|
| **Vanilla RNN** | Baseline architecture with simple RNN cells | Fixed-length context vector |
| **LSTM** | Improved with Long Short-Term Memory units | Better long-term dependency handling |
| **LSTM + Attention** | Advanced with Bahdanau attention mechanism | Dynamic source token focusing |

### 📊 **Dataset**: CodeSearchNet Python
- Source: [Nan-Do/code-search-net-python](https://huggingface.co/datasets/Nan-Do/code-search-net-python)
- Training samples: 8,000
- Validation samples: 1,000
- Test samples: 1,000
- Max docstring length: 50 tokens
- Max code length: 80 tokens

---

## 🚀 **Quick Start (5 Minutes)**

### **Option 1: Generate Attention Plots Only (Fastest)**
```bash
# If models are already trained
python src/main.py --model attention --skip-training --evaluate --plot --save-results
```

### **Option 2: Train Attention Model Only**
```bash
# Train just the attention model (3-4 hours)
python src/main.py --model attention --evaluate --plot --save-results
```

### **Option 3: Train All Models (8-10 hours)**
```bash
# Train all three models overnight
python src/main.py --train-all --evaluate --plot --save-results
```

---

## 📦 **Installation**

### **1. Clone Repository**
```bash
git clone https://github.com/2753Farhan/Seq2Seq
cd Seq2Seq
```

### **2. Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only:
# pip install torch torchvision torchaudio

pip install numpy matplotlib seaborn pandas nltk datasets tqdm pyyaml jupyter
python -c "import nltk; nltk.download('punkt')"
```

### **4. Download Dataset**
```bash
python scripts/download_data.py
```

---

## 🎮 **Command Line Arguments**

| Flag | Description | Example |
|------|-------------|---------|
| `--train-all` | Train all three models | `python src/main.py --train-all` |
| `--model` | Train specific model (`vanilla`/`lstm`/`attention`) | `python src/main.py --model attention` |
| `--skip-training` | Skip training, load pre-trained models | `python src/main.py --model attention --skip-training` |
| `--evaluate` | Evaluate models on test set | `python src/main.py --model attention --evaluate` |
| `--plot` | Generate plots (loss curves, attention heatmaps) | `python src/main.py --model attention --plot` |
| `--save-results` | Save results to files | `python src/main.py --model attention --save-results` |

---

## 🏃 **Common Usage Scenarios**

### **Scenario 1: Complete Project Run (All Models)**
```bash
# Train all models, evaluate, generate plots, save results
python src/main.py --train-all --evaluate --plot --save-results
```

### **Scenario 2: Focus on Attention Model**
```bash
# Train only attention model
python src/main.py --model attention --evaluate --plot --save-results
```

### **Scenario 3: Generate Attention Plots Only (No Training)**
```bash
# Use this if models are already trained (takes 2-3 minutes)
python src/main.py --model attention --skip-training --evaluate --plot --save-results
```

### **Scenario 4: Evaluate All Pre-trained Models**
```bash
# Load all trained models and evaluate without retraining
python src/main.py --train-all --skip-training --evaluate --plot --save-results
```

### **Scenario 5: Quick Test (Small Dataset)**
Edit `config/config.yaml` first:
```yaml
data:
  train_size: 1000  # Reduced from 8000
  valid_size: 200
  test_size: 200
training:
  epochs: 3  # Reduced from 15
```
Then run:
```bash
python src/main.py --model attention --evaluate --plot --save-results
```

---

## 📁 **Project Structure**

```
Seq2Seq/
├── config/
│   └── config.yaml              # Configuration parameters
├── src/
│   ├── data/
│   │   ├── dataset.py           # Dataset loading
│   │   └── tokenizer.py         # Vocabulary and tokenization
│   ├── models/
│   │   ├── encoder.py           # Base encoder (RNN/LSTM/Bidirectional)
│   │   ├── vanilla_rnn.py       # Vanilla RNN decoder and Seq2Seq
│   │   ├── lstm.py              # LSTM decoder and Seq2Seq
│   │   └── attention.py         # Bahdanau attention and decoder
│   ├── training/
│   │   ├── trainer.py           # Training loop with early stopping
│   │   └── evaluator.py         # Evaluation metrics
│   └── utils/
│       ├── metrics.py           # BLEU, token accuracy
│       └── visualization.py     # Plots and attention heatmaps
├── scripts/
│   ├── download_data.py         # Dataset download
│   └── test_dataset.py          # Dataset verification
├── outputs/
│   ├── models/                   # Saved model checkpoints
│   ├── plots/                     # Generated visualizations
│   └── results/                   # Evaluation results
├── main.py                        # Main execution script
└── requirements.txt               # Dependencies
```

---

## 📊 **Output Files**

After running, check these directories:

### **1. Model Checkpoints** (`outputs/models/`)
```
vanilla.pt              # Trained Vanilla RNN
lstm.pt                  # Trained LSTM
attention.pt             # Trained Attention model
attention_best.pt        # Best checkpoint
loss_history_*.json      # Training/validation loss history
```

### **2. Plots** (`outputs/plots/`)
```
loss_comparison.png      # Training/validation loss curves
metric_comparison.png    # BLEU and token accuracy comparison
bleu_vs_len.png          # Performance vs docstring length
attention_example_1.png  # Verified attention heatmap 1
attention_example_2.png  # Verified attention heatmap 2
attention_example_3.png  # Verified attention heatmap 3
attention_verification.txt  # Verification log
```

### **3. Results** (`outputs/results/`)
```
results.txt              # Numerical results summary
metrics.json             # Detailed metrics in JSON
predictions_eval_*.csv   # Sample predictions with BLEU scores
attention_examples.txt   # Verified examples for report
```

### **4. Logs**
```
training.log             # Complete training log
```

---

## ⏱️ **Expected Runtime (with GPU)**

| Operation | Time |
|-----------|------|
| Training all three models (15 epochs) | 8-10 hours |
| Training attention model only | 3-4 hours |
| Training LSTM only | 2-3 hours |
| Training Vanilla RNN only | 1-2 hours |
| Generating attention plots (skip-training) | 2-3 minutes |
| Full evaluation on test set | 5-10 minutes |

---

## 🔧 **Configuration Options**

Edit `config/config.yaml` to customize:

```yaml
# Data Configuration
data:
  train_size: 8000      # Number of training examples
  valid_size: 1000      # Validation examples
  test_size: 1000       # Test examples
  batch_size: 32        # Batch size
  max_src_len: 50       # Max docstring tokens
  max_tgt_len: 80       # Max code tokens

# Model Configuration
model:
  embedding_dim: 256    # Embedding size
  hidden_dim: 512       # Hidden state size
  num_layers: 2         # Number of RNN layers
  dropout: 0.5          # Dropout rate
  teacher_forcing_ratio: 0.5  # Initial teacher forcing

# Training Configuration
training:
  epochs: 15            # Max epochs
  learning_rate: 0.001  # Learning rate
  weight_decay: 1e-5    # L2 regularization
  early_stopping_patience: 3  # Stop if no improvement
```

---

## 🐛 **Troubleshooting**

### **Issue 1: CUDA Out of Memory**
```bash
# Reduce batch size in config.yaml
data:
  batch_size: 16  # Instead of 32
```

### **Issue 2: Dataset Download Fails**
```bash
# Manual download fallback
python scripts/download_data.py --force
# OR use synthetic dataset (automatic fallback)
```

### **Issue 3: Attention Plots Wrong**
The code includes verification:
```bash
# Check verification log
cat outputs/plots/attention_verification.txt
# Shows which source/target pairs were used
```

### **Issue 4: Slow Training**
```bash
# Reduce dataset size in config.yaml
data:
  train_size: 4000  # Half the size
training:
  epochs: 8         # Fewer epochs
```

---

## 📝 **Report Generation**

After all models are trained, generate the final report:

```bash
# Run this to create report/REPORT.md
python make_report.py
```

The report includes:
- All numerical results
- Loss curves
- Attention visualizations
- Error analysis
- Performance vs length analysis

---

## ✅ **Verification Checklist**

Run these commands to verify everything works:

```bash
# 1. Test dataset loading
python scripts/test_dataset.py

# 2. Quick attention plot generation (2-3 minutes)
python src/main.py --model attention --skip-training --plot

# 3. Full evaluation (5-10 minutes)
python src/main.py --model attention --skip-training --evaluate --plot --save-results

# 4. Check outputs
ls outputs/plots/
ls outputs/results/
```