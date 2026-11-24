# PAMR: Hybrid Evidence Scoring for Source-Session Retrieval in Agent Memory

Research artifact for the DASFAA 2027 submission *Provenance-Aligned Multi-Resolution Session Retrieval for Compressed Agent Memory*.

**PAMR** ranks past conversation sessions by how well they support a user's query. It uses a fixed dense encoder and a small, trained linear comparator. An LLM verifies links between turns offline. Both online retrieval stages run on CPU without calling an LLM.

## Why PAMR?

Suppose a user asks, “Which hotel did I end up booking?” A planning session might mention three hotels. A later session might contain the booking confirmation in a single turn. Both sessions match the topic, but the confirmation identifies which hotel the user booked.

- **Whole-session encoding** compresses a conversation into one vector. A short confirmation can have little influence on that vector.
- **Topic-summary retrieval** can identify the trip while giving the same topic score to several sessions. The summary alone cannot tell the reader where the confirmation occurred.
- **Online LLM reranking** can inspect the candidate conversations, but adds inference time and cost to each query.

PAMR keeps each turn tied to its source session. The initial ranking uses both the turn and its topic context. A verified link can then bring a related session into the reranking pool, where the comparator scores its own evidence.

```text
Offline    Source conversations
           ├─ Retained user turns and local windows, grouped by source session
           ├─ Topic summaries mapped to their contributing sessions
           ├─ Session and turn BM25 indexes
           └─ LLM-verified links between turns in different sessions

Online     Query + prepared index
             ↓
           Stage 1   Turn similarity + topic context + session BM25
             ↓       Initial ranking of source sessions
           Stage 2   One-hop graph nominations + 11-feature linear comparator
             ↓       Bounded swaps at ranks 1, 2–5, 6–10 and 11–20
           Final ranking of source sessions
```

## Quick Start

Run these commands from the repository root with Python 3.11. The example retrieves from prepared text using cached MiniLM vectors. The benchmark starts from the bundled Stage 1 features and turn graphs, refits the comparator, and evaluates session rankings on LoCoMo, LongMemEval-S, and LongMemEval-M.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python retrieve.py
python evaluate.py --model both --output outputs
python -m unittest discover -s tests -v
```

`outputs/per-query.csv` records the ranks of the first and last labeled evidence sessions for each query. `outputs/metrics.json` records fold sizes, training-pair counts, comparator weights, and the following metrics, expressed as percentages.

- **Success@k** counts queries with at least one labeled evidence session in the top *k*.
- **Recall@k** counts queries with all labeled evidence sessions in the top *k*. A missing candidate still counts as missing evidence.
- **MRR** averages the reciprocal rank of the first labeled evidence session. Queries with no retrieved evidence contribute zero.

Success and Recall are evaluated at *k* = 1, 5, and 10.

## How PAMR Works

### Offline indexing

Each retained user turn carries a source session ID and a position within that session. Local windows contain the center turn and its immediate neighbors in the same session. Each topic summary lists the sessions that contributed to it. Two BM25 indexes support lexical matching, one over concatenated turns per session and one over individual turns.

The offline verifier links turns from different sessions when their text supports the same matter or related activities of the same participant. At query time, the retriever can follow each verified link in either direction.

### Stage 1 · Hybrid initial retrieval

PAMR combines three scores for each session with indexed user turns.

| Signal | Computation | Default |
| --- | --- | --- |
| Turn similarity | Pool query–turn cosine similarities within the session using normalized smooth-max. | Temperature τ = 0.05 |
| Topic context | Take the highest query similarity among summaries mapped to the session. A session with no mapped summary uses its turn score. | Topic weight β = 0.10 within the semantic score |
| Session BM25 | Score the concatenated user turns and normalize by the highest session BM25 score for this query. | Lexical weight α = 0.25 |

The initial score is `(1 − α) × ((1 − β) × turn + β × topic) + α × BM25`.

### Stage 2 · Graph-proposed reranking

1. **Rank turns.** Weighted reciprocal rank fusion (RRF) combines dense turn ranks with ranks from positive BM25 matches. The rank offset is 60 and the lexical weight is 0.5.
2. **Nominate sessions.** Each of the top 20 initial sessions supplies up to two linked anchor turns in RRF order. Each anchor nominates the sessions of its three highest-ranked one-hop neighbors, or all its neighbors if it has fewer than three.
3. **Score the evidence.** A linear comparator trained by pairwise logistic regression scores each session. Its 11 normalized features cover the initial score, turn and local-window matches, keyword coverage, feature interactions, and the best fused turn rank.
4. **Apply bounded swaps.** Visit rank segments 1, 2–5, 6–10, and 11–20 in order. From below each segment, take up to six eligible sessions in rank order, then up to two additional graph-nominated sessions. Fill any remaining slots in the eight-session pool by rank. The best challenger swaps with the weakest eligible session in the segment if their comparator scores differ by at least `ln(1.5)` in the challenger's favor. Each segment allows one swap and both participants retire from further swaps.

## Running on Custom Text

Use [examples/history.json](examples/history.json) as the input format for retained turns, topic summaries and their source sessions, verified turn links, and queries. The example supplies its own summaries and links. For a new history, prepare the compressed turns, summaries, and verified links upstream.

The bundled demo uses cached vectors for three queries. To encode your own text, install the encoder dependencies, substitute your JSON path for `examples/history.json`, and point `--model` to a local copy of [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

```sh
python -m pip install -r requirements-encoder.txt
python retrieve.py --history examples/history.json --model /path/to/all-MiniLM-L6-v2
```

To reuse the embeddings, save them on the first run and pass the cache on subsequent runs. The retriever checks a digest of the ordered input texts before using the cache.

```sh
python retrieve.py --history examples/history.json --model /path/to/all-MiniLM-L6-v2 --save-vectors history.npz
python retrieve.py --history examples/history.json --vectors history.npz
```

## Evaluation Protocols

| Option | Comparator training | Evaluation queries |
| --- | --- | --- |
| `--model transfer` | Fit on the six-conversation LoCoMo development split (969 queries). Apply the same weights to S and M. | Four held-out LoCoMo conversations (567 queries) and the matched 400 queries in each LongMemEval variant. |
| `--model fivefold` | Use the same LoCoMo split. Within S and M, train five comparators separately per dataset, each on the eligible queries from the other folds. | The LoCoMo holdout and one held-out prediction per query in each 400-query LongMemEval cohort. |
| `--model both` | Run both protocols and fit the common LoCoMo model once. | Report the LoCoMo holdout once and each LongMemEval protocol separately. |

Five-fold training excludes held-out groups, groups with overlapping evidence, and abstention examples. Groups account for original questions and their variants. The matched evaluation cohort includes 21 abstention variants in each LongMemEval dataset, scored on retrieval of their reference sessions.

To use the historical fold population of 500 questions per LongMemEval dataset, pass `--fold-population archive`. Evaluation still uses the matched 400 per dataset. The swap margin stays fixed across protocols.

## Building the Shared Graph for LongMemEval-M

`build_m_graph.py` pools source conversations from the 500 M histories into one SQLite graph. It accepts the official JSON array or a directory with one `HISTORY_ID.json` file per history. Histories share a session record when both its ID and content match. The builder uses session IDs, dates, roles, and conversation text. Question IDs identify histories for visibility checks. Question text, answers, and evidence labels are excluded from indexing.

```sh
# Propose pairs from M source histories in a new database.
python build_m_graph.py propose longmemeval_m_cleaned.json outputs/m.sqlite3

# Import positive M judgments from the upstream verifier.
python build_m_graph.py apply outputs/m.sqlite3 verified_m_judgments.jsonl

# Inspect a seed session's neighbors in one history.
python build_m_graph.py neighbors outputs/m.sqlite3 HISTORY_ID SEED_SESSION

# Use the visible edges when retrieving from a prepared M history.
python retrieve.py --history prepared_m_history.json --model /path/to/all-MiniLM-L6-v2 \
  --m-graph outputs/m.sqlite3 --m-history HISTORY_ID
```

- **Candidate selection.** The builder shortlists turns from different sessions whose personal statements share rare terms. Both sessions must occur in chronological order in at least one witness history. The `candidates` table stores a key bound to the source turns, their endpoint IDs, the witness history, and the lexical score.
- **Upstream verification.** The research pipeline filters candidate pairs using Qwen `text-embedding-v4` at cosine similarity ≥ 0.40. Among the highest-scoring 256 pairs, `gpt-4o-mini-2024-07-18` judges those with eligible evidence spans. The public `apply` command takes the resulting positive judgments as a JSONL input.
- **Judgment format.** Each row supplies `candidate`, `origin` (`M-shared`), `model` (`gpt-4o-mini-2024-07-18`), `request_id`, `cosine`, `relation` (`same_matter` or `activity_history`), and `old` and `new` spans. Each span is `[start_offset, exclusive_end_offset, exact_quote]` in the original turn. The importer checks the candidate and both quotes. Re-importing an identical judgment keeps one edge.
- **Query-time visibility.** An edge is visible when both endpoints occur in the selected history in the verified order. Prepared turns match by source session ID and original text. An optional zero-based `source_position` distinguishes repeated turns. All queries in the prepared history share one projected adjacency list.

## Repository & Benchmark Files

| File | Contents |
| --- | --- |
| [retrieve.py](retrieve.py) | Builds an index from prepared text and runs both retrieval stages for each query. |
| [evaluate.py](evaluate.py) | Refits comparators on stored Stage 1 inputs, recomputes turn-level nominations, applies bounded swaps, and aggregates metrics. It also scores the earlier session-level nominations for a paired comparison and checks historical transfer ranks. |
| [build_m_graph.py](build_m_graph.py) | Proposes M candidates, imports verified judgments, and projects visible neighbors for a given history. |
| [method/](method/) | Implements scoring, graph operations, candidate nomination, training, and reranking. |
| `data/*.npz` | Stores 11-column session features in Stage 1 rank order, evidence masks, and historical session-level nominations. |
| `data/LoCoMo.json`, `data/S.json`, `data/M.json` | Map public question IDs to candidate rows, gold counts, folds, group conflicts, and evaluation cohorts. Session indices are local to each query. |
| `data/*-graph.json.gz` | Stores linked turns as `(initial_session_index, full_history_dense_rank, positive_bm25_rank)` triples, with zero for a missing BM25 match. Edges refer to those triples. Isolated turns are omitted after computing ranks and queries with no reachable seed edge may have an empty graph. |
| [data/comparator.json](data/comparator.json) | Contains the ordered feature names and fitted LoCoMo weights used by the retrieval example. |

The bundled `data/M-graph.json.gz` comes from the historical shared graph, which combines judgments obtained in M with records projected from S. The shared-graph commands above build from M sources alone. Original conversations and encoder weights are available from their maintainers. [LoCoMo](https://github.com/snap-research/locomo) · [LongMemEval](https://github.com/xiaowu0162/LongMemEval) · [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Acknowledgments & License

The BM25 and pooling primitives are adapted from LightMem. Its MIT notice is included in [LICENSE](LICENSE).
