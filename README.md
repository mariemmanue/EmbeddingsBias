# Bias and Retrieval Experiments (BBQ)

This section of the repository documents experiments using the **Bias Benchmark for QA (BBQ)** dataset to evaluate how identity-bearing language shifts semantic similarity, retrieval, and generation in large language models (LLMs) and embedding models.

---

## Research Questions

1. **RQ1 – Volatility:**  
   How unstable are representations or predictions across equally ambiguous stereotype contexts?

2. **RQ2 – Disambiguation Gain:**  
   Does added evidence improve alignment or accuracy asymmetrically for stereotype-consistent vs. inconsistent cases?

3. **RQ3 – Polarity Effect:**  
   Are negative questions more semantically or behaviorally aligned with stereotypes than their non-negative complements?



---

## Input Data

- **Dataset:** [`heegyu/BBQ`](https://huggingface.co/datasets/heegyu/BBQ) (Bias Benchmark for QA)
- **Metadata file:** `BBQ/additional_metadata.csv`

The BBQ dataset contains question–answer pairs covering 11 social categories (e.g., Age, Gender Identity, Race/Ethnicity, Religion). Each example is accompanied by metadata identifying its category and bias condition (stereotyped, anti-stereotyped, or neutral).

---

## Expected Outputs

Each experiment produces structured results under the directory passed to `--output-dir` (for example, `runs/BBQ_gen_gemma1b_full/`).  
The files in each run folder capture different stages of processing and analysis:

### 1. Merged and Filtered CSVs (`*_rows.csv`)
These CSVs contain the complete row-level data used for analysis.  
Each row corresponds to one context-question–answer pair from the BBQ dataset,
---
### 2. Per-Research-Question Metrics (`*_rq*.csv`)
Each file summarizes results aligned with a specific research question:

| File | Description | Research Question |
|------|--------------|-------------------|
| `*_rq1_similarity.csv` | Average cosine similarity between neutral and identity-injected embeddings, per model and category. | **RQ1:** Representation shift |
| `*_rq2_retrieval.csv` | Retrieval rank changes or document overlap scores across query variants. | **RQ2:** Retrieval variation |
| `*_rq3_volatility_gen.csv` | Generation-based volatility metrics (variance in log-probabilities or completions) grouped by category and bias condition. | **RQ3:** Generative bias / semantic instability |
---

### 3. Visualizations (`*_volatility.png`, etc.)
PNG figures are automatically generated to illustrate key bias metrics and model behaviors.

**Typical plots include:**
- **Volatility plots (`*_volatility.png`)** – show model uncertainty or instability across categories.  
  - *X-axis:* category (e.g., Age, Race, Religion)  
  - *Y-axis:* volatility metric (e.g., variance in log-probs or cosine distance)  
  - *Interpretation:* higher volatility indicates that identity-bearing text produces larger representational or generative shifts.

- **Bias direction plots (`*_bias_score.png`)** – visualize the direction and magnitude of stereotyped vs. anti-stereotyped model responses.  
  - Bars > 0 → model favors the stereotyped answer; Bars < 0 → model favors the anti-stereotyped one.

- **Embedding similarity heatmaps (`*_similarity_heatmap.png`)** – show pairwise cosine similarities across categories, indicating how strongly different identity cues cluster together in embedding space.

---

## Running the BBQ Experiments

### 1. **Setup**

Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-org-or-user>/EmbeddingsBias.git
cd EmbeddingsBias
pip install -r requirements.txt
```

### 2. Generative Experiments

These experiments test whether generative models exhibit biased reasoning or differential text production across identity conditions. They compute model log-probabilities and completions for each BBQ question context.

### Example A — GPU (Gemma 1B, full dataset)

```bash
python BBQ/BBQ_Experiments.py \
  --task generative \
  --dataset-id heegyu/BBQ \
  --metadata-csv BBQ/additional_metadata.csv \
  --output-dir runs/BBQ_gen_gemma1b_full \
  --device cuda \
  --dtype float16 \
  --use-logprobs \
  --gen-model google/gemma-3-1b-it
```



### 3. Embedding Experiments

Embedding experiments measure how identity-bearing text changes semantic representations by comparing cosine similarity between neutral and identity-injected variants.

Example A — GPU (Qwen 4B embedding):

```bash
python BBQ/BBQ_Experiments.py \
  --task embedding \
  --dataset-id heegyu/BBQ \
  --metadata-csv BBQ/additional_metadata.csv \
  --output-dir runs/BBQ_emb_qwen \
  --device cuda \
  --dtype float16 \
  --batch-size 64 \
  --embedding-models Qwen/Qwen3-Embedding-4B
  ```
  
 ## Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

# Contact 
For questions or collaboration inquiries, contact Marie Tano (mtano@stanford.edu) or open an issue on this repository.
