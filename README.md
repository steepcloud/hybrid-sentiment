# Hybrid Sentiment Analysis

A comprehensive sentiment analysis framework combining deep learning encoders (LSTM, GRU, Transformer, BERT, RoBERTa, DistilBERT) with classical machine learning classifiers (Logistic Regression, Random Forest, XGBoost). Features both a Python CLI and a modern Tauri-based desktop GUI.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [GUI Application](#gui-application)
  - [Backend API](#backend-api)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a hybrid approach to sentiment analysis that combines the representational power of deep learning encoders with the efficiency and interpretability of classical machine learning classifiers.

**Two Training Paradigms:**

1. **Hybrid Approach**: Train deep learning encoder → Extract embeddings → Train classical ML classifier
2. **End-to-End**: Train deep learning model directly for classification

## Features

- **Multiple Deep Learning Encoders**:
  - LSTM (Bidirectional)
  - GRU (Bidirectional)
  - Transformer
  - BERT (bert-base-uncased)
  - RoBERTa (roberta-base)
  - DistilBERT (distilbert-base-uncased)

- **Classical ML Classifiers**:
  - Logistic Regression
  - Random Forest
  - XGBoost

- **Word Embeddings**:
  - Word2Vec (trained from scratch)
  - Support for pre-trained embeddings (GloVe, FastText)

- **Dual Interface**:
  - Command-line interface (CLI) for training and evaluation
  - Cross-platform desktop GUI built with Tauri + React
  - RESTful API with FastAPI

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - Statistical comparison between models
  - Training history visualization

## 🏗️ Architecture

### Hybrid Architecture
I'll help you create a comprehensive README for your hybrid sentiment analysis project. Let me first gather some information about your project structure and configuration. 

Read [](file:///d%3A/hybrid-sentiment/package.json)

Read [](file:///d%3A/hybrid-sentiment/requirements.txt)

Read [](file:///d%3A/hybrid-sentiment/configs/config.yaml#1-1), lines 1 to 100

Read [](file:///d%3A/hybrid-sentiment/src/main.py#1-1), lines 1 to 50

Now let me check if there's an existing README: 

Read [](file:///d%3A/hybrid-sentiment/README.md)

Perfect! Here's a comprehensive README for your project:

```markdown
# 🎭 Hybrid Sentiment Analysis

A comprehensive sentiment analysis framework combining deep learning encoders (LSTM, GRU, Transformer, BERT, RoBERTa, DistilBERT) with classical machine learning classifiers (Logistic Regression, Random Forest, XGBoost). Features both a Python CLI and a modern Tauri-based desktop GUI.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [GUI Application](#gui-application)
  - [Backend API](#backend-api)
- [Project Structure](#project-structure)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a hybrid approach to sentiment analysis that combines the representational power of deep learning encoders with the efficiency and interpretability of classical machine learning classifiers.

**Two Training Paradigms:**

1. **Hybrid Approach**: Train deep learning encoder → Extract embeddings → Train classical ML classifier
2. **End-to-End**: Train deep learning model directly for classification

## ✨ Features

- **Multiple Deep Learning Encoders**:
  - LSTM (Bidirectional)
  - GRU (Bidirectional)
  - Transformer
  - BERT (bert-base-uncased)
  - RoBERTa (roberta-base)
  - DistilBERT (distilbert-base-uncased)

- **Classical ML Classifiers**:
  - Logistic Regression
  - Random Forest
  - XGBoost

- **Word Embeddings**:
  - Word2Vec (trained from scratch)
  - Support for pre-trained embeddings (GloVe, FastText)

- **Dual Interface**:
  - Command-line interface (CLI) for training and evaluation
  - Cross-platform desktop GUI built with Tauri + React
  - RESTful API with FastAPI

- **Comprehensive Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - Statistical comparison between models
  - Training history visualization

## Architecture

### Hybrid Architecture
```bash
Text Input → Tokenization → Deep Encoder → Embeddings → Classical ML → Prediction
                ↓
         Word2Vec/BERT/etc.
```

### End-to-End Architecture
```bash
Text Input → Tokenization → Deep Model → Softmax → Prediction
```

## Installation

### Prerequisites

- Python 3.8+
- Node.js 16+ (for GUI)
- Rust (for GUI)
- CUDA-capable GPU (optional, for faster training)

### Python Backend

```bash
# Clone the repository
git clone https://github.com/steepcloud/hybrid-sentiment.git
cd hybrid-sentiment

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### GUI Application (Optional)

```bash
# Install Node dependencies
npm install

# Install Tauri CLI
npm install -g @tauri-apps/cli

# Run in development mode
npm run tauri dev

# Build production executable
npm run tauri build
```

## Usage

### CLI Interface

#### 1. Train Embeddings (Optional)
```bash
python src/main.py train-embeddings \
  --dataset imdb \
  --embedding word2vec \
  --vector_size 300 \
  --window 5
```

#### 2. Train Hybrid Model
```bash
# Train encoder
python src/main.py train-hybrid \
  --dataset imdb \
  --encoder lstm \
  --epochs 10 \
  --batch_size 32

# Train classifier on embeddings
python src/main.py train-classifier \
  --dataset imdb \
  --encoder lstm \
  --classifier xgboost
```

#### 3. Train End-to-End Model
```bash
python src/main.py train-e2e \
  --dataset imdb \
  --model bert \
  --epochs 3 \
  --batch_size 16
```

#### 4. Inference
```bash
python src/main.py predict \
  --encoder-path results/models/deep_learning/imdb/lstm/lstm_best.pt \
  --classifier-path results/models/classical_ml/imdb/lstm/xgboost.pkl \
  --text "This movie is absolutely amazing!"
```

#### 5. Evaluate and Compare Models
```bash
python src/main.py compare \
  --dataset imdb \
  --models lstm gru transformer bert
```

### GUI Application

Launch the desktop application:
```bash
npm run tauri dev
```

Features:
- Real-time sentiment prediction
- Model selection (Hybrid vs End-to-End)
- Confidence scores and probabilities
- Clean, modern interface

### Backend API

Start the FastAPI server:
```bash
python backend/app_backend.py
```

API endpoints:
- `POST /predict` - Single text prediction
- `POST /predict-batch` - Batch predictions
- `GET /models` - List available models

Example request:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Great movie!",
    "encoder_type": "lstm",
    "classifier_type": "xgboost"
  }'
```

## Project Structure

```
hybrid-sentiment/
├── backend/                  # FastAPI backend
│   └── app_backend.py
├── configs/                  # Configuration files
│   └── config.yaml
├── data/                     # Data directory
│   ├── raw/                  # Raw datasets
│   ├── processed/            # Preprocessed data
│   └── embeddings/           # Trained embeddings
├── notebooks/                # Jupyter notebooks
│   └── hybrid_sentiment_colab.ipynb
├── results/                  # Training results
│   ├── models/               # Saved models
│   ├── comparisons/          # Model comparisons
│   └── embeddings/           # Embedding visualizations
├── src/                      # Source code
│   ├── data/                 # Data loading & preprocessing
│   ├── models/               # Model implementations
│   │   ├── classical_ml/     # Logistic, RF, XGBoost
│   │   └── deep_learning/    # LSTM, GRU, Transformer, BERT
│   ├── training/             # Training scripts
│   ├── evaluation/           # Evaluation metrics
│   ├── visualization/        # Plotting utilities
│   └── main.py               # CLI entry point
├── src-tauri/                # Tauri (Rust) backend
├── src-ui/                   # React frontend
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── package.json              # Node.js dependencies
└── README.md
```

## Supported Models

### Deep Learning Encoders

| Model | Parameters | Embedding Dim | Best For |
|-------|-----------|---------------|----------|
| LSTM | 2-layer BiLSTM | 300 | Sequential patterns |
| GRU | 2-layer BiGRU | 300 | Faster than LSTM |
| Transformer | 3-layer, 6 heads | 300 | Long-range dependencies |
| BERT | 110M | 768 | State-of-the-art |
| RoBERTa | 125M | 768 | Robust pre-training |
| DistilBERT | 66M | 768 | Faster BERT variant |

### Classical ML Classifiers

- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method, handles non-linearity
- **XGBoost**: Gradient boosting, often best performance

## Datasets

Supported datasets:
- **IMDB Movie Reviews**: 50k movie reviews (25k train, 25k test)
- **Twitter Sentiment140**: Tweet sentiment analysis
- **Custom**: Add your own CSV dataset

## Results

### IMDB Movie Reviews Dataset

Performance comparison on IMDB test set (25,000 reviews):

#### Top Performing Models

| Model | Approach | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|----------|---------|
| **RoBERTa (E2E)** | End-to-End | **94.24%** | **0.9425** | **0.9895** |
| **BERT (E2E)** | End-to-End | 93.04% | 0.9294 | 0.9769 |
| **DistilBERT (E2E)** | End-to-End | 92.44% | 0.9241 | 0.9706 |
| **RoBERTa + Logistic** | Hybrid | 90.48% | 0.9048 | 0.9653 |
| **LSTM (E2E)** | End-to-End | 88.56% | 0.8859 | 0.9299 |

#### Hybrid vs End-to-End Comparison

**BERT-based Models:**
- BERT (E2E): 93.04% accuracy
- BERT + Logistic Regression: 86.19% accuracy
- BERT + Random Forest: 82.18% accuracy
- BERT + XGBoost: 77.68% accuracy

**Transformer-based Models:**
- Transformer (E2E): 81.96% accuracy
- Transformer + Logistic Regression: 69.06% accuracy
- Transformer + Random Forest: 64.10% accuracy
- Transformer + XGBoost: 61.39% accuracy

**LSTM/GRU Models:**
- LSTM (E2E): 88.56% accuracy
- GRU (E2E): 50.00% accuracy (failed training)
- LSTM + Logistic Regression: 56.65% accuracy
- GRU + Logistic Regression: 57.72% accuracy

### Twitter Sentiment140 Dataset

Performance comparison on Twitter test set:

#### Top Performing Models

| Model | Approach | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|----------|---------|
| **RoBERTa (E2E)** | End-to-End | **93.37%** | **0.9488** | **0.9804** |
| **BERT (E2E)** | End-to-End | 92.35% | 0.9423 | 0.9697 |
| **DistilBERT (E2E)** | End-to-End | 91.02% | 0.9318 | 0.9558 |
| **RoBERTa + Logistic** | Hybrid | 88.05% | 0.9091 | 0.9489 |
| **LSTM (E2E)** | End-to-End | 85.50% | 0.8898 | 0.8978 |

#### Hybrid vs End-to-End Comparison

**BERT-based Models:**
- BERT (E2E): 92.35% accuracy
- BERT + Logistic Regression: 86.18% accuracy
- BERT + Random Forest: 81.99% accuracy
- BERT + XGBoost: 75.74% accuracy

**Transformer-based Models:**
- Transformer (E2E): 83.25% accuracy
- Transformer + Logistic Regression: 69.60% accuracy
- Transformer + Random Forest: 65.42% accuracy
- Transformer + XGBoost: 64.93% accuracy

**LSTM/GRU Models:**
- LSTM (E2E): 85.50% accuracy
- GRU (E2E): 64.96% accuracy
- LSTM + Logistic Regression: 65.56% accuracy
- GRU + Logistic Regression: 65.95% accuracy

### Key Insights

1. **End-to-End Training Wins**: Transformer-based models (BERT, RoBERTa, DistilBERT) achieve best performance when trained end-to-end rather than using the hybrid approach.

2. **RoBERTa is Top Performer**: RoBERTa (E2E) achieves the highest accuracy on both datasets:
   - IMDB: 94.24% accuracy with 0.9895 ROC-AUC
   - Twitter: 93.37% accuracy with 0.9804 ROC-AUC

3. **BERT Models Excel**: All BERT variants (BERT, RoBERTa, DistilBERT) significantly outperform traditional RNN-based models (LSTM, GRU, Transformer).

4. **Hybrid Approach Trade-offs**: 
   - Hybrid models are faster to train (train encoder once, then quick classical ML)
   - End-to-end models achieve 5-10% higher accuracy but require more training time
   - Best hybrid: RoBERTa + Logistic Regression (90.48% IMDB, 88.05% Twitter)

5. **DistilBERT Efficiency**: DistilBERT provides excellent performance (92.44% IMDB, 91.02% Twitter) with 40% fewer parameters than BERT, making it ideal for production deployments.

6. **Classical ML Classifier Ranking** (for hybrid models):
   - Logistic Regression: Best performance, fastest training
   - Random Forest: Good balance, handles non-linearity
   - XGBoost: Lower performance in this setting, possibly due to embedding feature space

### Training Times (approximate, on GPU)

| Model | IMDB Training Time | Twitter Training Time |
|-------|-------------------|---------------------|
| BERT (E2E) | ~60 min | ~90 min |
| RoBERTa (E2E) | ~65 min | ~95 min |
| DistilBERT (E2E) | ~40 min | ~60 min |
| LSTM (E2E) | ~15 min | ~25 min |
| Transformer (E2E) | ~20 min | ~30 min |
| LSTM + XGBoost (Hybrid) | ~10 min + 2 min | ~15 min + 3 min |

*Note: Times measured on NVIDIA RTX GPU. CPU training takes 5-10x longer.*

## Configuration

Edit config.yaml to customize:

```yaml
# Dataset selection
data:
  dataset_name: "imdb"
  max_length: 256
  vocab_size: 20000

# Model architecture
deep_learning:
  lstm:
    hidden_dim: 128
    num_layers: 2
    dropout: 0.3
  
  bert:
    model_name: "bert-base-uncased"
    learning_rate: 2e-5
    batch_size: 16

# Training parameters
training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.001
  early_stopping_patience: 3
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.