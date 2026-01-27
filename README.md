# Emirical Study of Decoder-Only Language Model Fine-Tuning for Low-Resource QA

### Overview

This repository contains an empirical, controlled study of fine-tuning decoder-only language models for question–answering (QA) tasks in a low-resource setting.

We fine-tune and evaluate multiple pretrained decoder-only LLMs on the same small QA dataset (200 QA pairs) using a shared training protocol, enabling direct comparison of model behavior, generalization, and overfitting characteristics.

The goal of this project is not to achieve state-of-the-art performance, but to analyze and understand how different model architectures behave under identical, limited supervision.

### Models Evaluated

The following models were fine-tuned using the same QA format, loss masking strategy, and training pipeline:
- GPT-2
- Phi-2 (2.7B)
- LLaMA-2-7B
- Mistral-7B-v0.1
  
All models were trained using LoRA (Low-Rank Adaptation) to ensure parameter-efficient fine-tuning and reproducibility.

---

### Dataset

Task: Question Answering (QA)

Dataset size: 200 question–answer pairs
```
Format:

Q: <question>
A: <answer>
```

Setting: Low-resource / small-data regime

This setup reflects practical scenarios where large labeled datasets are unavailable, such as domain-specific academic or industrial applications.

---

### Methodology

Training Setup

**Model type**: Decoder-only causal language models

**Fine-tuning strategy**: LoRA (parameter-efficient fine-tuning)

**Loss**: Causal language modeling loss with prompt loss masking

Question tokens are used as conditioning context

Loss is applied only to answer tokens

Batching: Custom padding and masking to support variable-length sequences

Evaluation: Fixed validation split with qualitative and loss-based analysis

All hyperparameters, trainable parameter counts, and loss curves are logged for each experiment.

---

Evaluation & Analysis

We evaluate models using:

Training loss vs validation loss trends
Qualitative QA generations
Overfitting behavior
Prompt sensitivity
Stability and robustness of generated answers

Given the open-ended nature of QA, emphasis is placed on comparative qualitative analysis rather than automated metrics alone.

### Key Observations (Summary)

Smaller models overfit more quickly in low-resource settings

Larger models exhibit better robustness and consistency

Prompt sensitivity varies significantly across architectures

Model scale is not the only predictor of QA performance under limited supervision

Detailed results and examples are provided in the experiment logs.

---
Reproducibility

All experiments are fully logged

Hyperparameters and trainable parameter counts are recorded

The same training and evaluation pipeline is used for all models

Only LoRA adapters are trained (base model weights are unchanged)
