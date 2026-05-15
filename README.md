# GPT From Scratch

A from-scratch implementation of a GPT-style Transformer Language Model using PyTorch.

---

## Features

- GPT-style Transformer architecture
- Multi-Head Causal Self Attention
- Feed Forward Networks & Layer Normalization
- Autoregressive Next-Token Prediction
- Custom Dataset & Dataloader Pipeline
- Training Loop with:
  - Cross Entropy Loss
  - AdamW Optimizer
  - Validation Evaluation
  - Model Checkpointing
- Text Generation with:
  - Greedy Decoding
  - Temperature Sampling
  - Top-k Sampling
- GPT-2 Tokenization using Tiktoken (BPE)

---

## Project Structure

```text
attention.py      -> Attention implementations
transformer.py    -> Transformer building blocks
mini_GPT.py       -> GPT model architecture
dataset.py        -> GPT dataset & dataloader
train.py          -> Training pipeline
generate.py       -> Text generation pipeline
```

---

## Project Goal

The goal of this project is to deeply understand how GPT-style Large Language Models work internally by implementing the core components from scratch instead of relying entirely on high-level frameworks.

---

## Learning Resources

- **Vizuara — Building LLMs from Scratch**  
  https://youtube.com/playlist?list=PLPTV0NXA_ZSgsLAr8YCgCwhPIJNNtexWu

- **Andrej Karpathy — NanoGPT**  
  https://youtu.be/kCc8FmEb1nY