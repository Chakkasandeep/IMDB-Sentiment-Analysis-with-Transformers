# 🎬 IMDB Sentiment Analysis with Transformers

A complete implementation of a transformer-based model for sentiment analysis on the IMDB movie review dataset. This project builds a custom GPT-style transformer from scratch using PyTorch to classify movie reviews as positive or negative.

## 📊 Project Overview

This project demonstrates:
- Building a transformer model from scratch (no pre-trained models)
- Sentiment classification on IMDB dataset (50K reviews)
- Achieving **77.38%** accuracy on test set
- Complete pipeline from data loading to model evaluation

## 🔄 Workflow
<img width="400" height="600" alt="image" src="https://github.com/user-attachments/assets/d33997f7-c697-4e3f-8114-069734ee5d40" />


## 🎯 Results

| Metric | Value |
|--------|-------|
| Training Samples | 22,500 |
| Validation Samples | 2,500 |
| Test Samples | 25,000 |
| **Final Test Accuracy** | **77.38%** |
| Training Time | ~4 epochs |

### Training Progress
- Epoch 1: 72.28% validation accuracy
- Epoch 2: 77.04% validation accuracy
- Epoch 3: 78.20% validation accuracy
- Epoch 4: 79.08% validation accuracy

## 🏗️ Model Architecture

### DemoGPT Configuration
```python
{
    "vocabulary_size": 30522,
    "num_classes": 2,
    "d_embed": 128,
    "context_size": 128,
    "layers_num": 4,
    "heads_num": 4,
    "head_size": 32,
    "dropout_rate": 0.1
}
```

### Architecture Components
1. **Embedding Layer**: Token + Positional embeddings (128-dim)
2. **Transformer Blocks** (4 layers):
   - Multi-head attention (4 heads × 32-dim)
   - Feed-forward network (128 → 512 → 128)
   - Layer normalization
   - Residual connections
   - Dropout (0.1)
3. **Classification Head**: Mean pooling + Linear layer (2 classes)

## 📋 Requirements

```bash
pip install torch torchvision
pip install transformers
pip install pandas matplotlib seaborn
```

### Dependencies
- Python 3.8+
- PyTorch 1.9+
- Transformers (Hugging Face)
- Pandas
- Matplotlib
- Seaborn

## 📂 Dataset Structure

```
aclImdb/
├── train/
│   ├── pos/     # 12,500 positive reviews
│   └── neg/     # 12,500 negative reviews
└── test/
    ├── pos/     # 12,500 positive reviews
    └── neg/     # 12,500 negative reviews
```

Download the IMDB dataset from: [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/chakkasandeep/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

### 2. Download Dataset
```bash
# Download and extract IMDB dataset
wget http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
```

### 3. Run the Training Script
```bash
python sentiment_analysis.py
```

## 📝 Code Structure

The project follows a modular structure:

### Main Components

1. **Data Loading** (`load_dataset`)
   - Reads text files from directories
   - Creates pandas DataFrames
   - Splits into train/validation/test sets

2. **Tokenization** (`IMDBDataset`)
   - Uses BERT tokenizer (bert-base-uncased)
   - Truncates/pads to 128 tokens
   - Converts to PyTorch tensors

3. **Model Components**
   - `AttentionHead`: Single attention head with causal masking
   - `MultiHeadAttention`: Combines multiple attention heads
   - `FeedForward`: Position-wise feed-forward network
   - `Block`: Complete transformer block
   - `DemoGPT`: Full model architecture

4. **Training Loop**
   - AdamW optimizer (lr=3e-4)
   - CrossEntropyLoss
   - Batch size: 32
   - 4 training epochs

5. **Evaluation** (`calculate_accuracy`)
   - Computes accuracy on validation/test sets
   - No-gradient evaluation mode

## 🔑 Key Features

### ✨ Custom Transformer Implementation
- Built from scratch using PyTorch primitives
- Causal attention masking
- Layer normalization and residual connections
- Dropout for regularization

### 📊 Comprehensive EDA
- Label distribution visualization
- Review length analysis
- Sample review inspection

### 🎯 Classification Adaptation
- Mean pooling across sequence
- Binary classification head
- Efficient batch processing

## 📈 Performance Metrics

```python
# Training Configuration
EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
MAX_LENGTH = 128
```

## 🧪 Validation

The code includes multiple assertion checks:
- Dataset dimensions verification
- Tensor shape validation
- Model output verification
- Accuracy calculation validation

## 📊 Visualizations

The script generates:
1. `label_distribution.png` - Distribution of positive/negative reviews
2. `review_length_distribution.png` - Character count distribution

## 🔍 Sample Predictions

```python
# Positive Review Example
"This movie was absolutely fantastic! Great acting and storyline."
→ Prediction: Positive (1)

# Negative Review Example
"Terrible waste of time. Poor acting and boring plot."
→ Prediction: Negative (0)
```

## 🎓 Learning Outcomes

This project demonstrates:

1. **Subword Tokenization**: BERT tokenizer balances vocabulary size and representation
2. **Classification Adaptation**: Transformers modified for sequence classification
3. **Training Best Practices**: Proper train/val/test splits and evaluation
4. **Transformer Versatility**: Effective beyond text generation tasks


##  Acknowledgments

- IMDB Dataset: [Andrew Maas et al.](http://ai.stanford.edu/~amaas/data/sentiment/)
- BERT Tokenizer: [Hugging Face Transformers](https://huggingface.co/transformers/)

---
