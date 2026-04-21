#Official implementation of the paper:

**Language-Based Personality Assessment from Life Narratives: A Focus on Model Interpretability and Efficiency**  
(*Frontiers in Artificial Intelligence*)

Paper: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1760246/abstract

---

## Overview

This repository provides code for predicting Big Five personality traits from long-form life narrative interviews.

The work addresses key challenges in this setting:
- handling transcripts that exceed standard transformer input limits  
- improving interpretability of model predictions  
- maintaining computational efficiency  

The approach uses a two-step framework that combines transformer-based representations with sequential modeling.

---

## Method

The proposed framework consists of two stages:

### 1. Segment-Level Encoding  
A pretrained transformer model (e.g., RoBERTa-large) is fine-tuned using a sliding window approach over long transcripts. Each window produces a representation of a segment of the narrative.

### 2. Sequence-Level Aggregation  
Segment-level embeddings are aggregated using:
- an RNN with attention (primary model)  
- a feedforward network (ablation)  

This allows the model to capture long-range dependencies and identify the most informative parts of the narrative.

---

## Repository Structure

```
src/
  extract_embeddings.py
  train_roberta_*.py
  train_longformer_neuroticism.py
  train_llama.py
  run_topic_modeling.py

notebooks/
  *_finetune_*.py
  *_rnn_*.py
  *_ffn_*.py
  embeddings_exploration.py
```

---

## Data

The dataset used in this project is confidential and is not included in this repository.

This repository does not contain:
- raw transcript text  
- participant-level labels  
- derived embeddings  
- trained model checkpoints  
- experimental outputs  

To run the code, users must provide their own data in a compatible format and ensure they have the appropriate permissions.

### Expected format

The scripts assume data in JSON format with fields such as:
- `PARTID`: participant identifier  
- `text`: transcript text  

Additional label files (e.g., CSV) should include the relevant personality trait columns used in the scripts.

---

## Usage

A typical workflow consists of:

### 1. Fine-tune the transformer
```
python src/train_roberta_*.py
```

### 2. Extract embeddings
```
python src/extract_embeddings.py
```

### 3. Train aggregation model
```
python notebooks/openness_rnn_final.py
```

### 4. (Optional) Interpretability analysis
```
python src/run_topic_modeling.py
```

---

## Requirements

Install dependencies with:

```
pip install -r requirements.txt
```

Core dependencies include:
- PyTorch  
- transformers  
- scikit-learn  
- simpletransformers  
- optuna  
- BERTopic  

---

## Notes

- The codebase reflects experimental pipelines used in the paper and may include multiple variants of similar models.
- Paths to data and outputs may need to be adapted to your local setup.

---

## Citation

If you use this repository, please cite the associated Frontiers in Artificial Intelligence paper.
