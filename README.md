# BBQ Dataset information
## Overview
BBQ has two key dimensions:
- Context condition:
    - Ambiguous: context missing key info; correct answer should be “unknown” (the UNKNOWN option).
    - Disambiguated: extra sentence tells you which person is actually the answer.
- Question polarity:
    - Negative question: “Who is greedy / violent / bad at math?” etc.
    - Non-negative question: “Who is generous / peaceful / good at math

Experimental conditions
- Generative model: ARC vs RACE formatting (“raceprompt” vs “arcprompt” scoring in run_generative.py)
- Embedding experiments: qavscontext, questionvscontext, answervsquestion (EXPTYPES in run_embeddings.py).


## Generative: ARC vs RACE prompts
From run_generative.py, every row has:
​- promptarc, predarc, accarc: ARC-style question formatting.
- promptrace, predrace, accrace: RACE-style formatting.
- goldlabel: the correct answer letter 
- sclabel: stereotype-consistent answer.
- silabel: anti-stereotype answer.
- contextcondition3: AMBIG, DISAMBIGSTEREO, DISAMBIGANTI, DISAMBIG.

---

## Input Data

- **Dataset:** [`heegyu/BBQ`](https://huggingface.co/datasets/heegyu/BBQ) (Bias Benchmark for QA)
- **Metadata file:** `BBQ/additional_metadata.csv`

The BBQ dataset contains question–answer pairs covering 11 social categories (e.g., Age, Gender Identity, Race/Ethnicity, Religion). Each example is accompanied by metadata identifying its category and bias condition (stereotyped, anti-stereotyped, or neutral).
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
