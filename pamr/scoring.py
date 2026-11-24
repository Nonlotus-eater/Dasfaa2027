"""Session scoring primitives adapted from LightMem. See LICENSE."""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

Array = np.ndarray
TOKEN_PATTERN = re.compile(r"[\w]+", flags=re.UNICODE)


def _as_matrix(vectors: Array) -> Array:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    return matrix


def normalize_rows(vectors: Array) -> Array:
    """Return row-wise L2-normalized vectors without modifying the input."""
    matrix = _as_matrix(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.finfo(np.float32).eps)


@dataclass(frozen=True)
class BM25Index:
    """Query-independent BM25 statistics for a fixed set of sessions."""

    term_frequencies: dict[str, Counter[str]]
    document_lengths: dict[str, int]
    document_frequency: Counter[str]
    average_length: float


def build_bm25_index(
    session_texts: Mapping[str, Sequence[str]],
) -> BM25Index:
    """Build reusable document statistics for complete source sessions."""
    documents = {
        str(session_id): TOKEN_PATTERN.findall(" ".join(texts).lower())
        for session_id, texts in session_texts.items()
    }
    term_frequencies = {
        session_id: Counter(document) for session_id, document in documents.items()
    }
    document_frequency: Counter[str] = Counter()
    for frequencies in term_frequencies.values():
        document_frequency.update(frequencies.keys())
    return BM25Index(
        term_frequencies=term_frequencies,
        document_lengths={
            session_id: len(document) for session_id, document in documents.items()
        },
        document_frequency=document_frequency,
        average_length=(
            sum(map(len, documents.values())) / len(documents) if documents else 0.0
        ),
    )


def score_bm25_index(
    query_text: str,
    index: BM25Index,
    *,
    k1: float = 1.2,
    b: float = 0.75,
) -> dict[str, float]:
    """Score a query against a prebuilt session BM25 index."""
    if k1 <= 0.0 or not 0.0 <= b <= 1.0:
        raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
    if not index.term_frequencies:
        return {}
    query_tokens = set(TOKEN_PATTERN.findall(query_text.lower()))
    scores: dict[str, float] = {}
    document_count = len(index.term_frequencies)
    for session_id, frequencies in index.term_frequencies.items():
        length_norm = (
            1.0
            - b
            + b
            * index.document_lengths[session_id]
            / max(index.average_length, 1.0)
        )
        score = 0.0
        for token, term_frequency in frequencies.items():
            if token not in query_tokens:
                continue
            frequency = index.document_frequency[token]
            inverse_frequency = math.log(
                1.0 + (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            score += inverse_frequency * (
                term_frequency * (k1 + 1.0) / (term_frequency + k1 * length_norm)
            )
        scores[session_id] = score
    maximum = max(scores.values(), default=0.0)
    return {
        session_id: (score / maximum if maximum > 0.0 else 0.0)
        for session_id, score in scores.items()
    }


def score_session_vectors(
    query_vector: Array,
    session_vectors: Mapping[str, Array],
    *,
    pooling: str = "max",
    temperature: float = 0.05,
) -> dict[str, float]:
    """Score session multi-vectors with one of the supported pooling rules.

    ``smooth_max`` subtracts ``temperature * log(n)`` from log-sum-exp.  It is
    bounded by the maximum cosine score and discounts a lone accidental match
    when the rest of a session provides no support.

    ``softmax_weighted_average`` uses ``softmax(score / temperature)`` as the
    turn weights and returns their similarity-weighted average.  ``max`` is a
    backward-compatible alias for the explicitly named ``hard_max``.
    """
    supported_pooling = {
        "max",
        "hard_max",
        "smooth_max",
        "softmax_weighted_average",
    }
    if pooling not in supported_pooling:
        raise ValueError(
            "pooling must be 'max', 'hard_max', 'smooth_max', or "
            "'softmax_weighted_average'"
        )
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    query = normalize_rows(np.asarray(query_vector).reshape(1, -1))[0]
    result: dict[str, float] = {}
    for session_id, vectors in session_vectors.items():
        scores = normalize_rows(vectors) @ query
        maximum = float(np.max(scores))
        if pooling in {"max", "hard_max"}:
            result[str(session_id)] = maximum
        elif pooling == "smooth_max":
            shifted = (scores - maximum) / temperature
            result[str(session_id)] = float(
                maximum + temperature * np.log(np.exp(shifted).mean())
            )
        else:
            shifted = (scores - maximum) / temperature
            weights = np.exp(shifted)
            weights /= weights.sum()
            result[str(session_id)] = float(np.dot(weights, scores))
    return result
