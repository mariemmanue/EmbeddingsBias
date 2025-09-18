# Bias and Retrieval Experiments (BBQ + NQ)

This repository contains code and results for a series of experiments probing whether identity-bearing language shifts semantic similarity, retrieval, and generation in large language models (LLMs) and embedding models.

---

## Repository Layout

```text
IBM2025/
├── BBQ4.py                   # Python file for Experiment 4 (RQ4 on the BBQ dataset, entity position)
├── NQ_RQ1.py                 # Python file for RQ1 experiment on the Natural Questions (NQ) dataset
├── NQ_RQ3.py                 # Python file for experiments RQ2–RQ3 on the Natural Questions (NQ) dataset
├── BBQ/                      # Folder containing BBQ1_3.ipynb file, and doc listing extracted BBQ entities
    ├── BBQ1_3.ipynb          # Python notebook for experiments 1–3 (RQ1–RQ3 on the BBQ dataset)
├── NQ/                       # Results for NQ experiments
├── granite/                  # Granite embedding CSV/Parquet
├── rq2g_vocab_preview.csv    # List of BBQ extracted entities
├── requirements.txt
├── README.md
└── Final Presentation.key    # Keynote internship presentation (old draft)
```
 
---

## Research Questions

| Experiment ID | Research Question |
|---|---|
| **BBQ-RQ1** | Is there variability in cosine similarity across ambiguous contexts for the same question? | 
| **BBQ-RQ2a** | Does disambiguation increase cosine similarity with the question? | 
| **BBQ-RQ2b** | Does disambiguation increase similarity more when it counters stereotypes? | 
| **BBQ-RQ3** | Do negative questions align more with stereotype-confirming contexts than non-negative ones? |
| **BBQ-RQ4** | Does entity position (first vs second mention) affect similarity to the question? |
| **NQ-RQ1** | Does injecting an identity into an answer passage shift its similarity to a neutral question? Is the shift identity- or position-dependent?  |
| **NQ-RQ2** | Does identity injection in prompts/queries affect retrieval/ranking? |
| **NQ-RQ3** | Does identity injection in prompts/queries affect generated outputs? |

---


## Input Data

- **BBQ**: Loaded via HuggingFace (`heegyu/bbq`, multiple subsets).
- **NQ**: Loaded via HuggingFace (`sentence-transformers/natural-questions`).
- **Granite embeddings**: CSV/Parquet files in repo.
- **Identity vocabulary**: `rq2g_vocab_preview.csv` with identity-bearing terms used for injection experiments.

---

## Expected Ouputs

- `BBQ1_3.ipynb` → cosine similarity tables, Δ plots, t-test/ANOVA outputs
- `BBQ4.py`   → Experiment 4 (RQ4 on the BBQ dataset, entity position)
- `NQ_RQ1.py` → RQ1 Experiment on the Natural Questions (NQ) dataset
- `NQ_RQ3.py` → Experiments on the Natural Questions (NQ) dataset (RQ2–RQ3)



---

## Running the Experiments

### BBQ4.py — CLI for RQ4 (Entity Position)
python BBQ4.py \
    --input out/granite.parquet\
    --output-dir  \
    --model ibm-granite/granite-embedding-30m-english \
    --batch-size\
    --sample-n  \
    --embed-model ibm-granite/granite-embedding-30m-english\

### NQ_RQ1.py — CLI for RQ1 (Identity Injection/Similarity Shift)
python NQ_RQ1.py \
    --input granite/granite_embeddings.csv \
    --output-dir  \
    --model ibm-granite/granite-embedding-30m-english \
    --batch-size  \
    --text-col-question question \
    --embed-batch-size\
    --begin-and-end \
    --pairs-per-question \
    --embed-max-tokens\

### NQ_RQ3.py — CLI for RQ2 and RQ3 (Reranking and generation)
python NQ_RQ3.py \
    --rq3-nq-jsonl (json file with NQ positives and negative matches)\
    --output-dir  \
    --model ibm-granite/granite-embedding-30m-english \
    --text-col-question question \
    --embed-batch-size\
    --gen-batch-size\
    --begin-and-end \
    --pairs-per-question \
    --embed-max-tokens\
    --rq3-topk\
    --rq3-per-category\
    --rq3-gen-tokens\
    --rq3-gen-model\
    --rq3-rits-key $RITS_API_KEY\

---


## Dependencies
Install requirements:
```bash
pip install -r requirements.txt
