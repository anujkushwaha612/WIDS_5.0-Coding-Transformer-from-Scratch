# Simple GPT Model

A lightweight implementation of a GPT (Generative Pre-trained Transformer) language model built from scratch using PyTorch. This project demonstrates the core concepts of transformer architecture for text generation.

## Overview

This implementation includes:

- Multi-head self-attention mechanism
- Transformer blocks with residual connections and layer normalization
- Positional embeddings
- Character-level tokenization
- Text generation capabilities


## Features

- **Transformer Architecture**: Built with attention heads, feed-forward networks, and layer normalization
- **Multi-Head Attention**: Parallel attention mechanisms for better representation learning
- **Residual Connections**: Skip connections to improve gradient flow
- **Text Generation**: Generate new text based on learned patterns
- **GPU Support**: Automatic CUDA detection and utilization


## Model Architecture

- **Embedding Dimension**: 384
- **Number of Heads**: 6
- **Number of Layers**: 6
- **Batch Size**: 15
- **Block Size**: 60 (context length)
- **Dropout Rate**: 0.2
- **Parameters**: ~10.8M


## Requirements

```
torch 
```


## Setup

1. Prepare the training data in a file named `input.txt` in the same directory
2. The model will automatically create character-level vocabulary from the text

## Usage

### Training

```bash
python gpt.py
```

The model will:

1. Load and preprocess text from `input.txt`
2. Train for 9,999 iterations with periodic evaluation
3. Generate sample text after training

### Configuration

Key hyperparameters can be modified at the top of the file:

```python
BATCH_SIZE = 15      # Batch size for training
BLOCK_SIZE = 60      # Context window size
LEARNING_RATE = 1e-4 # Learning rate
N_EMBED = 384        # Embedding dimension
N_HEAD = 6           # Number of attention heads
N_LAYER = 6          # Number of transformer layers
DROPOUT = 0.2        # Dropout rate
```


## Model Components

### AttentionHead

Implements scaled dot-product attention with causal masking for autoregressive generation.

### MultiHeadAttention

Combines multiple attention heads and applies a projection layer.

### FeedForwardNetwork

Simple MLP with ReLU activation and dropout.

### TransformerBlock

Complete transformer block with self-attention, feed-forward network, and residual connections with layer normalization.

### SimpleGPTModel

Main model class that combines token embeddings, positional embeddings, transformer blocks, and the language modeling head.

## Training Process

- **Data Split**: 90% training, 10% validation
- **Loss Function**: Cross-entropy loss
- **Optimizer**: AdamW
- **Evaluation**: Periodic loss evaluation on both training and validation sets


## Text Generation

The model generates text autoregressively by:

1. Starting with a context (can be empty)
2. Predicting the next token based on current context
3. Appending the predicted token to context
4. Repeating until desired length is reached

## Output

During training:

- Model parameter count
- Periodic training and validation losses
- Generated text sample after training completion


## Notes

- The model uses character-level tokenization
- Includes causal masking to prevent looking at future tokens
- Implements proper weight initialization following GPT practices
- Uses pre-layer normalization (Pre-LN) architecture


## Customization

To adapt this model for other use case:

1. Replace `input.txt` with the training data
2. Adjust hyperparameters based on the dataset size and computational resources
3. Modify the generation length and sampling strategy as needed


