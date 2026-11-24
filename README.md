# PAMR session retrieval

PAMR ranks the source sessions that contain evidence for a conversational query. This artifact provides a retrieval example with prepared memory, a trainable pairwise comparator, and the inputs for re-evaluating 567 held-out LoCoMo questions and the same 400 questions in each LongMemEval variant.

## Run locally

Use Python 3.11 and run these commands in the extracted repository.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python retrieve.py
python evaluate.py --model both --output outputs
python -m unittest discover -s tests -v
```

The example and the benchmark run on CPU with the files included here. The benchmark writes per-query first and last evidence ranks and a JSON summary with fold sizes, training-pair counts and fitted weights. Success@k requires at least one labeled session in the top k. Recall@k requires every labeled session, including labels absent from the candidate set. MRR averages the reciprocal rank of the first labeled session. All three metrics are expressed as percentages.

## Retrieval inputs

The example takes retained user turns, their source session and chronological position, topic summaries with their contributing sessions, and verified undirected links between turns. It constructs session and turn BM25 indexes, pools turn scores with normalized smooth-max (temperature 0.05), and mixes topic and session BM25 scores with weights 0.10 and 0.25. Each local window covers the center turn and its immediate neighbors within its session.

At query time, dense turn ranks and positive BM25 ranks are fused using weighted reciprocal ranks (offset 60, lexical weight 0.5). Each of the first 20 sessions contributes up to two linked anchors. Each anchor contributes up to three best-ranked neighbors. For each rank segment ending at 1, 5, 10 or 20, up to six direct candidates and two graph-priority candidates enter the pool. Remaining slots follow the initial ranking. A swap requires a difference of at least ln(1.5) and retires both participants.

The three example queries use cached MiniLM vectors. To supply new text, prepare a JSON file following `examples/history.json` and point the encoder to a local copy of all-MiniLM-L6-v2.

```sh
python -m pip install -r requirements-encoder.txt
python retrieve.py --history examples/history.json --model /path/to/all-MiniLM-L6-v2
```

Optionally supply `--save-vectors history.npz` with the encoder, then use `--vectors history.npz` with the same input. The cache checks a digest of the ordered input texts. Topic formation, turn compression and LLM judgments are upstream inputs to the portable scripts. The supplied example uses authored summaries and links. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## One shared graph for LongMemEval-M

The M builder pools the source conversations from the 500 histories into one SQLite database. The public LongMemEval-M array and a directory of individual M histories are both supported. Records with the same source session ID and identical content share a session. The shortlist uses rare terms shared by personal statements from different sessions. Each proposed pair must have a witness history in which its earlier and later sessions occur in that order. Question text, answers and evidence labels are excluded from the index.

```sh
python build_m_graph.py propose longmemeval_m_cleaned.json outputs/m.sqlite3
python build_m_graph.py apply outputs/m.sqlite3 verified_m_judgments.jsonl
python build_m_graph.py neighbors outputs/m.sqlite3 HISTORY_ID SEED_SESSION
python retrieve.py --history prepared_m_history.json --model /path/to/all-MiniLM-L6-v2 --m-graph outputs/m.sqlite3 --m-history HISTORY_ID
```

The `candidates` table contains the source-bound candidate key, endpoint turn IDs, witness history and lexical score. In the research run, Qwen text-embedding-v4 filters the shortlist at cosine 0.40 and GPT-4o-mini judges eligible span options among the highest-scoring 256 pairs. The portable `apply` command imports positive judgments already verified by that upstream process. Each JSONL row carries `candidate`, `origin` (`M-shared`), `model` (`gpt-4o-mini-2024-07-18`), `request_id`, `cosine`, `relation` (`same_matter` or `activity_history`), and `old` and `new` evidence spans. Each span is an array of the start offset, exclusive end offset and exact quote in the original turn. The importer checks membership and both quotes. Re-importing the same judgment is safe.

At retrieval time the history ID selects visible edges from this database. Both endpoints must be present in chronological order. Prepared turns match using their source session IDs and original text. An optional zero-based `source_position` identifies a repeated turn within a session. The retriever builds the projected adjacency list once for all queries in that prepared history.

## Evaluation protocols

| Option | Training population | Reported queries |
| --- | --- | --- |
| `--model transfer` | Refit the comparator on 969 questions from six LoCoMo conversations | Four disjoint LoCoMo conversations (567 questions) and 400 each in S and M |
| `--model fivefold` | Same LoCoMo holdout. For S and M, fit five models **separately per dataset** using the other folds of the 400-question cohort | One held-out prediction per question |
| `--model both` | Run both protocols, fitting the shared LoCoMo model once | Separate metrics for each protocol |

The five-fold path retains predefined groups across original questions, variants and shared evidence. It excludes held-out groups, groups with overlapping evidence, and abstention examples from training. The 400-question evaluation cohort includes 21 abstention variants in each LongMemEval dataset. These questions are scored on retrieval of their reference sessions. To reproduce the historical training population, add `--fold-population archive`. This fits folds over 500 questions and still reports the matched 400. The margin is fixed in all runs.

## Corrected retrieval results

The M numbers below and the bundled `data/M-graph.json.gz` reproduce the historical experiment. That shared graph included 1,682 verification records projected from S and 207 obtained in M. The M-only builder above has been exercised on real M histories. Its full-cohort retrieval metrics are pending.

| Dataset / training | N | Success@1 | Success@5 | Success@10 | Recall@1 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LoCoMo holdout | 567 | 73.19 | 92.59 | 95.77 | 60.32 | 79.89 | 86.42 | 81.25 |
| LongMemEval-S / transfer | 400 | 89.50 | 97.50 | 98.50 | 33.75 | 89.50 | 94.75 | 92.93 |
| LongMemEval-S / five folds | 400 | 90.00 | 98.00 | 98.25 | 34.25 | 89.25 | 94.75 | 93.30 |
| LongMemEval-M / transfer | 400 | 73.75 | 90.75 | 94.25 | 26.00 | 75.75 | 84.75 | 81.38 |
| LongMemEval-M / five folds | 400 | 74.25 | 91.00 | 94.25 | 26.50 | 75.00 | 83.75 | 81.91 |

## Benchmark files

- `data/*.npz` stores numeric 11-column normalized session features in Stage 1 rank order, evidence masks and historical session-level nominations. The JSON manifests identify public question IDs, candidate spans, total gold counts, folds, group conflicts, and the primary evaluation cohort. Session indices are local to each question.
- `data/*-graph.json.gz` stores aligned, linked turns as triples of initial session index, full-history dense rank, and positive BM25 rank (zero for a non-match). Edges index these triples. Isolated turns are omitted after computing full-history ranks. Queries without a reachable seed edge may have an empty graph. These inputs support the fixed Stage 1 ranking used by this benchmark.
- `evaluate.py` recomputes turn-level nominations, refits models, applies bounded swaps and aggregates held-out rankings. It also evaluates the earlier session-level nominations using identical scores for a paired comparison. Previous evidence ranks are used only to check the historical transfer control.
- The full-cohort evaluation starts from stored Stage 1 inputs. `retrieve.py` separately exposes live scoring, feature extraction and both ranking stages for prepared text. Benchmark source conversations and model weights for external encoders are distributed by their maintainers. [LoCoMo](https://github.com/snap-research/locomo) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval)

The BM25 and pooling primitives are adapted from LightMem. Its MIT notice is included in `LICENSE`.
