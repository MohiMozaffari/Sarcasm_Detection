# Sarcasm Detection

End-to-end sarcasm detection on the [danofer/sarcasm](https://www.kaggle.com/datasets/danofer/sarcasm) Reddit dataset using a progression of models from a TF-IDF baseline to a LoRA-fine-tuned DeBERTa transformer with retrieval-augmented ensembling.

## Quick start

Open **`sarcasm-detection.ipynb`** and run all cells top-to-bottom on a Kaggle GPU (T4 or better). The notebook is fully self-contained — no separate files or imports needed.

```
Attach dataset: danofer/sarcasm  →  /kaggle/input/
Enable GPU accelerator
Run All
```

Training both transformers on the full data needs more than one Kaggle GPU session, so the notebook runs in sessions. Each session stops training before `session_time_budget_hours` (11 h; Kaggle stops GPU sessions at 12 h), saves a checkpoint, and still writes its outputs. To continue, attach the previous session's output as an input and run all cells again: finished stages are reloaded and interrupted training resumes from its last checkpoint. The last cell prints `Pipeline complete.` when nothing is left.

---

## Results

Full-data run on Kaggle (2×T4), evaluated once on the held-out test split (151,496 comments, 50 % sarcastic).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision |
|-------|----------|-----------|--------|----|---------|----------------|
| TF-IDF + Logistic Regression | 0.722 | 0.739 | 0.686 | 0.712 | 0.795 | 0.801 |
| DeBERTa-v3 LoRA, context-free | 0.710 | 0.670 | 0.825 | 0.740 | 0.809 | 0.821 |
| DeBERTa-v3 LoRA, context-aware | 0.731 | 0.696 | 0.822 | 0.753 | 0.826 | 0.834 |
| Retrieval-augmented stacker | **0.745** | **0.709** | **0.830** | **0.765** | **0.839** | **0.846** |

- The thresholds were tuned for F1 on the validation split: 0.5 for TF-IDF and 0.355 for the other three models (the F1 curve is flat between about 0.33 and 0.37). ROC-AUC and average precision don't depend on the threshold. The low threshold trades precision for recall, which is why the context-free model has a higher F1 than TF-IDF but lower accuracy.
- Parent context helps. On the test split, the context-aware model is right where the context-free one is wrong for 13,402 comments, and the reverse happens for 10,172 (a net gain of 3,230).
- Most of the stacker's gain comes from combining TF-IDF with the context-aware model, which get the largest stacker weights. The retrieval features add little: kNN retrieval on its own reaches only 0.609 ROC-AUC.
- Both DeBERTa models trained the full 2 epochs (22,094 steps) without early stopping. Validation ROC-AUC was still rising at the end (context-aware: 0.825).
- Only about 0.8 % of comment pairs exceed 256 tokens, so the sliding window rarely comes into play.

An earlier full-data run (July 2026) produced models worse than the TF-IDF baseline (context-free ROC-AUC 0.741). Those models were discarded and retrained after three fixes:
- **Precision:** a T4 reports bf16 support through emulation, so training had run in pure bf16. It now uses FP16 mixed precision with FP32 weights.
- **Session budget:** the time budget now ends training before Kaggle's 12-hour limit.
- **Best-checkpoint selection:** the best checkpoint is now chosen by ROC-AUC instead of F1 at 0.5.

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
| Epochs | 2 (early stopping on validation ROC-AUC, patience=2) |
| Learning rate | 2e-4 |
| Batch size | 8 per device (×4 gradient accumulation) |
| Precision | BF16 where the GPU runs it natively (Ampere+), otherwise FP16 mixed precision with FP32 weights (T4) |
| Gradient checkpointing | Off (fits a T4; ~30 % faster) |
| In-training evaluation | 20 k-row stratified validation subset every 3000 steps; final metrics use the full splits |

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
| Configuration | `SarcasmConfig` — set `debug=True` for a 20 k-row smoke test; restores finished stages and checkpoints from an attached earlier session |
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

For `parent_comment + comment` pairs that exceed 256 tokens, the parent context is divided into overlapping windows (stride = 64, max 4 windows). Each window is concatenated with the full reply and forwarded through the model; the resulting logits are averaged to produce a single prediction per example. A reply that alone would fill the budget is truncated so every window keeps at least 32 parent tokens.

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

All random seeds are fixed at `42`. Set `deterministic=True` in `set_all_seeds()` for exact GPU reproducibility (slower). The results above took three Kaggle sessions on 2×T4, about 21 GPU hours in total. At normal speed each DeBERTa model needs about 7 h of training, but session speed varies: one session ran about 2.5× slower. See Quick start for how sessions resume.
