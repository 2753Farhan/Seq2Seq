# Seq2Seq Text-to-Python Code Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-supported-blue)](https://www.docker.com/)

## 📋 Assignment Overview
Implementation and comparison of three Seq2Seq models for generating Python code from natural language docstrings:
1. Vanilla RNN-based Seq2Seq
2. LSTM-based Seq2Seq
3. LSTM with Attention mechanism

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)
```bash
git clone https://github.com/YOUR_USERNAME/seq2seq-code-generation.git
cd seq2seq-code-generation
chmod +x scripts/train_all.sh
./scripts/train_all.sh