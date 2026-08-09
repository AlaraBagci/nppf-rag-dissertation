# NPPF RAG Pipeline — MSc Dissertation

Comparative study of Retrieval-Augmented Generation (RAG) architectures for UK planning-policy
question answering, over a corpus combining the National Planning Policy Framework (NPPF) and
the Coventry Local Plan (plus supporting SPDs, Neighbourhood Plans, and related documents).

MSc Dissertation, WMG, University of Warwick.

## Repository Structure

```
src/            Core library modules (ingestion, indexing, retrieval, generation, evaluation, graph retrieval)
scripts/        One-off build scripts (graph construction, golden-dataset generation, embedding fine-tuning)
experiments/    One script per experiment configuration (exp00–exp08) plus the multi-hop evaluation harness
data/           Document corpus (data/dataset/) and the golden evaluation / fine-tuning datasets
outputs/        Every experiment's results and RAGAS scores (JSON), chunk files (JSONL), knowledge graphs
requirements.txt
.env.example    Configuration template — copy to .env and fill in your own credentials
```

## What's Excluded From This Repo (and Why)

The following are regenerable from the code and are excluded to keep the repository a reasonable
size (they total ~8.7GB):

| Excluded | Size | How to regenerate |
|---|---|---|
| `venv/` | 5.8GB | `pip install -r requirements.txt` |
| `outputs/chroma_db/` | 453MB | Re-run any experiment script — it rebuilds the vector index automatically |
| `outputs/bge-planning-finetuned/` | 1.3GB | `python scripts/finetune_embeddings.py` (the original, leakage-affected model — kept only for before/after comparison, see below) |
| `outputs/bge-planning-finetuned-v2/` | 1.3GB | `python scripts/generate_finetune_dataset.py` then `python scripts/finetune_embeddings.py` |
| `.env` | — | Never committed — copy `.env.example` and fill in your own Azure credentials |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your own Azure endpoint/API key values
```

## Important Methodological Note: Embedding Model Leakage

An earlier version of the embedding fine-tuning procedure trained directly on the same 60-question
golden evaluation set later used to score every experiment — a train/test leakage flaw. This was
identified, diagnosed, and corrected: `bge-planning-finetuned-v2` is trained on a separate,
leakage-free 300-pair set (`scripts/generate_finetune_dataset.py`, with a hard assertion that halts
training on any detected chunk overlap with either evaluation set). **All results in `outputs/`
from Exp04b onward are computed with the corrected `-v2` model.** See the accompanying technical
report for the full account of this investigation.

## Reproducing the Pipeline

Run in this order:

```bash
# 1. Ingest and chunk the corpus (produces outputs/chunks.jsonl)
python -m experiments.exp00_phase0

# 2. Build the leakage-free fine-tuning set, then fine-tune the embedding model
python scripts/generate_finetune_dataset.py
python scripts/finetune_embeddings.py

# 3. Build the knowledge graphs
python scripts/build_graph.py         # regex-based graph
python scripts/build_graph_llm.py     # + GPT-4.1 typed triples

# 4. Run the experiments (each is independent once the above exist)
python -m experiments.exp00_no_preprocess
python -m experiments.exp01_baseline
python -m experiments.exp02_hybrid
python -m experiments.exp03_hierarchical
python -m experiments.exp04_domain_embeddings
python -m experiments.exp05_graph_rag
python -m experiments.exp06_hierarchical_tuned
python -m experiments.exp07_llm_graph_rag
python -m experiments.exp08_hierarchical_graph_rag

# 5. Multi-hop cross-document benchmark and diagnostics (requires 1–4 above)
python scripts/generate_multihop_dataset.py
python -m experiments.eval_multihop
python scripts/diagnose_multihop.py 5          # baseline funnel
python scripts/diagnose_multihop.py 8          # widened top-k
python scripts/diagnose_multihop.py 5 reserved # reserved-slot reranking
```

## Evaluation

Every experiment is scored with [RAGAS](https://github.com/explodinggpt/ragas) 0.1.21
(faithfulness, answer_relevancy, context_precision, context_recall) plus a custom
`citation_quality` metric (GPT-4.1 judge, checking source attribution — a gap in standard RAGAS
not covered for this domain). Judge model: GPT-4.1. Generator: Llama-4-Maverick-17B-128E-Instruct-FP8.

## Results

All RAGAS scores and full per-question results are in `outputs/*_ragas_scores.json` and
`outputs/*_results.json`. See the accompanying technical report(s) for the full write-up,
design rationale, and mapping to the dissertation's research objectives.
