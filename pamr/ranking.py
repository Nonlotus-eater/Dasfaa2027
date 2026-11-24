"""Turn-level candidate nomination and bounded session swaps."""
from collections import Counter, defaultdict
import math

import numpy as np


def turn_ranks(dense, lexical):
    """Rank all turns by cosine and the positive matches by BM25 (one-based)."""
    dense = np.asarray(dense)
    lexical = np.asarray(lexical)
    if dense.ndim != 1 or lexical.shape != dense.shape:
        raise ValueError("Expected one dense score and one BM25 score per turn")
    dense_order = np.argsort(-dense, kind="stable")
    dense_rank = np.empty(len(dense), dtype=int)
    dense_rank[dense_order] = np.arange(1, len(dense) + 1)
    positive = np.flatnonzero(lexical > 0)
    word_order = positive[np.argsort(-lexical[positive], kind="stable")]
    word_rank = np.zeros(len(dense), dtype=int)
    word_rank[word_order] = np.arange(1, len(word_order) + 1)
    return dense_rank, word_rank


def mixed_order(dense_rank, word_rank):
    """Weighted RRF. A zero BM25 rank makes no lexical contribution."""
    dense_rank, word_rank = np.asarray(dense_rank), np.asarray(word_rank)
    score = 1.0 / (60 + dense_rank) + np.where(word_rank > 0, 0.5 / (60 + word_rank), 0.0)
    return np.lexsort((dense_rank, -score))


def adjacency(owners, edges):
    adjacent = defaultdict(set)
    for a, b in edges:
        if owners[a] != owners[b]:
            adjacent[a].add(b)
            adjacent[b].add(a)
    return adjacent


def nominate(owners, dense_rank, word_rank, edges, seeds, anchors=2, neighbors=3):
    return nominate_indexed(owners, dense_rank, word_rank, adjacency(owners, edges), seeds, anchors, neighbors)


def nominate_indexed(owners, dense_rank, word_rank, adjacent, seeds, anchors=2, neighbors=3):
    """Return a vote per selected neighbor using the history's prebuilt adjacency list."""
    order = mixed_order(dense_rank, word_rank)
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(len(order))
    seeds = set(seeds)
    used, support = Counter(), Counter()
    for turn in order:
        owner = owners[turn]
        if owner in seeds and turn in adjacent and used[owner] < anchors:
            used[owner] += 1
            endpoints = sorted(adjacent[turn], key=positions.__getitem__)[:neighbors]
            support.update(owners[n] for n in endpoints)
    return support


def rerank(scores, nominated, ratio=1.5):
    """Return indices into Stage 1 order. A pair of swapped sessions is retired."""
    if ratio < 1 or not math.isfinite(ratio):
        raise ValueError("The odds threshold must be finite and at least one")
    order = list(range(len(scores)))
    used = set()
    start = 0
    for stop in (1, 5, 10, 20):
        inside = [p for p in range(start, min(stop, len(order))) if order[p] not in used]
        tail = [p for p in range(stop, len(order)) if order[p] not in used]
        start = stop
        if not inside or not tail:
            continue
        direct, rest = tail[:6], tail[6:]
        graph = [p for p in rest if order[p] in nominated][:2]
        pool = (direct + graph + [p for p in rest if p not in graph])[:8]
        loser = min(inside, key=lambda p: (scores[order[p]], -p))
        winner = max(pool, key=lambda p: (scores[order[p]], -p))
        a, b = order[loser], order[winner]
        if scores[b] - scores[a] >= math.log(ratio):
            used.update((a, b))
            order[loser], order[winner] = b, a
    return order
