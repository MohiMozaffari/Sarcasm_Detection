# Sarcasm Detection

End-to-end sarcasm detection on the [danofer/sarcasm](https://www.kaggle.com/datasets/danofer/sarcasm) Reddit dataset using a progression of models from a TF-IDF baseline to a LoRA-fine-tuned DeBERTa transformer with retrieval-augmented ensembling.

## Quick start

Open **`sarcasm_detection_unified.ipynb`** and run all cells top-to-bottom on a Kaggle GPU (T4 or better). The notebook is fully self-contained — no separate files or imports needed.

```
Attach dataset: danofer/sarcasm  →  /kaggle/input/
Enable GPU accelerator
Run All
```

---

## Models

| # | Model | Input | Notes |
|---|-------|-------|-------|
| 1 | TF-IDF + Logistic Regression | `comment` | Fast baseline, 100 k features, bigrams |
| 2 | DeBERTa-v3-base + LoRA | `comment` | Context-free ablation |
| 3 | DeBERTa-v3-base + LoRA | `parent_comment + comment` | **Primary model**, sliding-window inference |
| 4 | Retrieval-augmented stacker | All probs + kNN features | BGE-base + FAISS + LogReg stacker |

### LoRA configuration (Models 2 & 3)

| Parameter | Value |
|-----------|-------|
| Base model | `microsoft/deberta-v3-base` |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| Dropout | 0.05 |
| Target modules | `query_proj`, `key_proj`, `value_proj` |
| Modules to save | `classifier`, `pooler` |
| Epochs | 2 (early stopping, patience=2) |
| Learning rate | 2e-4 |
| Batch size | 8 per device (×4 gradient accumulation) |
| Precision | BF16 / FP16 auto-detected |

---

## Dataset split

Stratified 70 / 15 / 15 split on the full cleaned dataset (~1 M Reddit comments).

| Split | Purpose |
|-------|---------|
| Train (70 %) | Model training |
| Validation (15 %) | Threshold tuning, stacker training, early stopping |
| Test (15 %) | Final evaluation (touched once) |

---

## Notebook structure

| Cell section | What happens |
|---|---|
| Install dependencies | `pip install` all required packages |
| Shared utility definitions | All helper classes and functions defined inline |
| Configuration | `SarcasmConfig` — set `debug=True` for a 20 k-row CPU smoke test |
| Step 1 — Data | Load CSV, clean text, create splits, length diagnostics |
| Step 2 — TF-IDF | Train baseline on full training split |
| Steps 3 & 4 — DeBERTa LoRA | Both transformer models trained with PEFT LoRA on full data |
| Step 5 — Retrieval ensemble | BGE embeddings → FAISS index → k-NN features → stacker |
| Step 6 — LLM prompts | Zero-shot & few-shot prompt templates saved (no API call) |
| Final comparison | Sorted metrics table across all models |
| Save outputs | All artifacts zipped to `sarcasm_unified_outputs.zip` |

---

## Outputs

| Artifact | Description |
|----------|-------------|
| `*_split.csv / .parquet` | Train / validation / test splits |
| `tfidf_logreg_model.joblib` | Saved TF-IDF pipeline |
| `deberta_v3_base_context_free_lora_adapter/` | PEFT adapter (context-free model) |
| `deberta_v3_base_context_aware_lora_adapter/` | PEFT adapter (context-aware model) |
| `*_metrics.json` | Per-model metrics (accuracy, F1, ROC-AUC, …) |
| `metrics_comparison.csv` | All models side-by-side |
| `predictions_test.csv` | Combined test predictions from all models |
| `error_analysis.csv` | High-confidence errors sorted by confidence |
| `confusion_matrix_*.png` | Confusion matrix plots |
| `retrieval_augmented_predictions.csv` | Ensemble predictions |
| `llm_prompt_examples.csv` | Zero-shot & few-shot prompts for test examples |
| `llm_predictions_template.csv` | Fill this with LLM predictions to evaluate them |
| `sarcasm_unified_outputs.zip` | All of the above, zipped |

---

## Sliding-window inference (context-aware model)

For `parent_comment + comment` pairs that exceed 256 tokens, the parent context is divided into overlapping windows (stride = 64, max 4 windows). Each window is concatenated with the full reply and forwarded through the model; the resulting logits are averaged to produce a single prediction per example.

---

## Retrieval-augmented ensemble

1. **Embed** up to 50 k training examples with `BAAI/bge-base-en-v1.5` (falls back to TF-IDF if `sentence-transformers` is unavailable).
2. **Index** with FAISS (`IndexFlatIP` on L2-normalised embeddings = cosine similarity).
3. **Retrieve** k=5 nearest neighbours for each validation / test example.
4. **Stack** retrieval features (weighted kNN probability, top-1 score, score gap, …) with TF-IDF / context-free / context-aware probabilities.
5. **Calibrate** the stacker on the validation split only; evaluate once on the test split.

---

## Requirements

```
transformers>=4.38
peft>=0.9
accelerate>=0.28
datasets
evaluate
sentencepiece
safetensors
pyarrow
joblib
sentence-transformers
faiss-cpu          # or faiss-gpu
scikit-learn
pandas
numpy
matplotlib
seaborn
```

Install with:

```bash
pip install transformers accelerate peft datasets evaluate sentencepiece safetensors \
            pyarrow joblib sentence-transformers faiss-cpu
```

---

## Reproducing results

All random seeds are fixed at `42`. Set `deterministic=True` in `set_all_seeds()` for exact GPU reproducibility (slower). The full pipeline on a Kaggle T4 takes approximately 3–4 hours.
