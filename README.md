# Source-Session Retrieval for Agent Memory

Official artifact for our DASFAA 2027 submission.

Our work introduces a lightweight, two-stage retrieval method for long-term conversational agent memory. Given a user query, it ranks the **source conversation sessions** that contain supporting evidence—decoupling offline memory indexing and cross-session graph construction from fast online retrieval with no LLM calls at query time.

---

## Why This Work?

When an agent interacts with a user across dozens or hundreds of sessions, multiple sessions often touch on the same broad topic (e.g., discussing travel plans in one session and confirming a hotel booking in another). Existing retrieval strategies face two common limitations:

- **Whole-session encoding** compresses an entire conversation into a single vector, allowing surrounding small talk to dilute fine-grained evidence turns.
- **Cross-session topic summaries** merge related turns from different sessions into shared topics, spreading topic scores across all contributing sessions and obscuring which specific session holds the direct answer.

**Our method addresses these limitations in two lightweight online stages over precomputed offline structures:**

1. **Offline Preparation**: Keeps retained user turns, topic summaries, and lexical (BM25) statistics aligned to their source sessions, alongside a verified cross-session turn association graph (topic formation, turn compression, and LLM verification are upstream offline inputs).
2. **Online Stage 1 (Hybrid Initial Retrieval)**: Combines turn-level semantic matching, topic-summary context, and session-level BM25 to produce an initial session ranking.
3. **Online Stage 2 (Graph-Proposed Reranking)**: Follows 1-hop links in the offline turn graph from top-ranked sessions to nominate cross-session candidates, scores sessions with a pairwise-trained linear comparator, and applies thresholded swaps across consecutive rank segments under a fixed per-segment candidate budget.

```text
[Offline Preparation (Upstream Inputs)]
  Retained User Turns + Topic Summaries + Session/Turn BM25 Indexes + Verified Cross-Session Turn Graph

[Online Retrieval (Zero LLM Calls)]
  Query q ──► Stage 1: Hybrid Initial Retrieval (Turn Smooth-Max + Topic Context + Session BM25)
                │
                ▼ Initial Ranking
              Stage 2: 1-Hop Turn-Graph Nomination + 11-Feature Linear Comparator
                │
                ▼ Bounded Pool (up to 6 Direct + 2 Graph-Priority per Segment) & Thresholded Swaps (1 / 2–5 / 6–10 / 11–20)
              Final Ranked Source Sessions
```

---

## Quick Start

This repository provides a self-contained retrieval example with prepared memory and cached vectors, a trainable pairwise comparator, and precomputed inputs for re-evaluating **567 held-out LoCoMo questions** and **400 questions each in LongMemEval-S and LongMemEval-M**. All experiments use Python 3.11 with seed 0. We measure latency on a single **NVIDIA RTX 3080 (10 GB)** with batch size 1.

```sh
# 1. Create environment and install dependencies
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

# 2. Run the end-to-end retrieval example (uses cached MiniLM vectors)
python retrieve.py

# 3. Reproduce benchmark evaluation on LoCoMo, LongMemEval-S, and LongMemEval-M
python evaluate.py --model both --output outputs

# 4. Run unit tests
python -m unittest discover -s tests -v
```

`evaluate.py` writes per-query first and last evidence ranks and a JSON summary with fold sizes, training-pair counts, fitted weights, and retrieval metrics (all reported as percentages):

* **Success@k**: At least one labeled evidence session appears in the top $k$.
* **Recall@k**: Every labeled evidence session appears in the top $k$ (including labels absent from the candidate set).
* **MRR**: Mean reciprocal rank of the first labeled evidence session.

---

## Method Overview

### 1. Memory Inputs & Stage 1 (Hybrid Initial Retrieval)

The retriever takes retained user turns (with source session ID and chronological position), topic summaries with their contributing sessions, and verified undirected links between turns. It builds both session-level and turn-level BM25 indexes.

At query time, Stage 1 scores each session by combining:

* **Turn-level semantic score**: Pooled across turns in the session using normalized smooth-max (temperature $\tau = 0.05$).
* **Topic-summary score**: Mixed with weight $0.10$.
* **Session-level BM25 score**: Mixed with weight $0.25$.

### 2. Stage 2 (Graph-Proposed Reranking)

Starting from the Stage 1 ranking, Stage 2 refines session order in three steps:

1. **Turn-level Weighted RRF**: Fuses dense turn ranks and positive turn-level BM25 ranks using weighted reciprocal rank fusion (offset $60$, lexical weight $0.5$).
2. **1-Hop Turn-Graph Nomination**: Each of the top $20$ sessions contributes up to $2$ linked anchor turns, and each anchor contributes up to $3$ best-ranked neighbor turns to nominate candidate sessions.
3. **Linear Comparator & Consecutive Segment Swapping**: Extracts 11 normalized session features (including Stage 1 score, turn semantic/BM25 matches, and local-window similarity covering the center turn and its immediate neighbors within the session) and scores sessions with a pairwise-trained linear comparator. For each rank segment ending at `1`, `5`, `10`, and `20`, a fixed-budget candidate pool admits up to **6 direct-rank candidates and 2 graph-priority candidates** (remaining slots follow the initial ranking). A swap requires a score margin of at least $\ln(1.5)$ and retires both participants for the query.

---

## Running on Custom Text

The bundled `retrieve.py` demo uses cached vectors for three example queries. To run on your own text:

1. Prepare a JSON file following `examples/history.json` (topic formation, turn compression, and LLM judgments are upstream inputs).
2. Install the encoder dependency and point `--model` to a local copy of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2):

```sh
python -m pip install -r requirements-encoder.txt
python retrieve.py --history examples/history.json --model /path/to/all-MiniLM-L6-v2
```

Optionally cache encoded vectors with `--save-vectors` and reuse them with `--vectors` (the cache validates a digest of the ordered input texts):

```sh
python retrieve.py --history examples/history.json --model /path/to/all-MiniLM-L6-v2 --save-vectors history.npz
python retrieve.py --history examples/history.json --vectors history.npz
```

---

## Evaluation Protocols

`evaluate.py` supports the following protocols via `--model`:

| Option | Training Population | Reported Queries |
| --- | --- | --- |
| `--model transfer` | Refit the comparator on 969 questions from 6 LoCoMo conversations | 4 disjoint LoCoMo conversations (567 questions) and 400 each in LongMemEval-S and LongMemEval-M |
| `--model fivefold` | Same LoCoMo holdout; for S and M, fit 5 models **separately per dataset** using the other folds of the 400-question cohort | One held-out prediction per question |
| `--model both` | Run both protocols (fitting the shared LoCoMo model once) | Separate metrics for each protocol |

**Protocol notes:**

* The five-fold path preserves predefined groups across original questions, variants, and shared evidence, excluding held-out groups, overlapping-evidence groups, and abstention examples from training.
* Each 400-question LongMemEval cohort includes 21 abstention variants, scored on retrieval of their reference sessions.
* To reproduce the historical 500-question training population (while still reporting on the matched 400 questions), add `--fold-population archive`. The swap margin $\ln(1.5)$ is fixed in all runs.

---

## Shared Graph Builder for LongMemEval-M

`build_m_graph.py` pools source conversations from all 500 LongMemEval-M histories into one SQLite database (supporting both the public JSON array and a directory of individual histories). Records with the same source session ID and identical content share a session; question text, answers, and evidence labels are excluded.

```sh
# 1. Propose cross-session candidate pairs sharing rare terms in chronological witness histories
python build_m_graph.py propose longmemeval_m_cleaned.json outputs/m.sqlite3

# 2. Import upstream verified positive judgments (idempotent; validates membership and exact quotes)
python build_m_graph.py apply outputs/m.sqlite3 verified_m_judgments.jsonl

# 3. Inspect neighbors for a given history and seed session
python build_m_graph.py neighbors outputs/m.sqlite3 HISTORY_ID SEED_SESSION

# 4. Run live retrieval with the projected per-history adjacency list
python retrieve.py --history prepared_m_history.json --model /path/to/all-MiniLM-L6-v2 \
  --m-graph outputs/m.sqlite3 --m-history HISTORY_ID
```

* **Research run**: The `candidates` table stores the source-bound candidate key, endpoint turn IDs, witness history, and lexical score. In our research run, `Qwen text-embedding-v4` filters the shortlist at cosine $\ge 0.40$, and `gpt-4o-mini-2024-07-18` judges eligible span options among the highest-scoring 256 pairs.
* **JSONL format for `apply`**: Each row contains `candidate`, `origin` (`M-shared`), `model` (`gpt-4o-mini-2024-07-18`), `request_id`, `cosine`, `relation` (`same_matter` or `activity_history`), and `old`/`new` evidence spans formatted as `[start_offset, exclusive_end_offset, exact_quote]`. Re-importing the same judgment is safe.
* **Query-time projection**: Given `HISTORY_ID`, the retriever selects only edges whose endpoints are both present in chronological order. Prepared turns match via source session ID and original text (with an optional zero-based `source_position` for repeated turns within a session), and the projected adjacency list is built once per history.

---

## Repository & Benchmark Files

* `retrieve.py` — Live scoring, feature extraction, and both ranking stages for prepared text.
* `evaluate.py` — Recomputes turn-level nominations from stored Stage 1 inputs, refits models, applies bounded swaps, and aggregates held-out rankings (also evaluates earlier session-level nominations using identical scores for a paired comparison).
* `build_m_graph.py` — SQLite-backed shared graph builder and importer for LongMemEval-M.
* `data/*.npz` & `data/*.json` — 11-column normalized session features in Stage 1 rank order, evidence masks, historical session-level nominations, and manifests (public question IDs, candidate spans, total gold counts, folds, group conflicts, and primary evaluation cohort).
* `data/*-graph.json.gz` — Aligned, linked turns stored as `(initial_session_index, full_history_dense_rank, positive_bm25_rank)` triples (zero for a non-match; isolated turns omitted after computing full-history ranks; `data/M-graph.json.gz` combines verification records obtained in M with records projected from S).

Benchmark source conversations and external encoder weights are distributed by their maintainers: [LoCoMo](https://github.com/snap-research/locomo) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval) · [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

---

## Acknowledgments & License

The BM25 and pooling primitives are adapted from LightMem. Its MIT notice is included in `LICENSE`.
