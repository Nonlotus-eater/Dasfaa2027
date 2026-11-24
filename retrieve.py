"""Retrieve sessions from prepared memory using cached or locally encoded MiniLM vectors."""
import argparse
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import numpy as np

from method.retrieval import Index, embedding_texts
from method.m_graph import retrieval_edges

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE / "examples" / "history.json"


def run(history_path, model=None, vector_path=None, save_vectors=None, *, m_graph=None, m_history=None):
    history = json.loads(history_path.read_text())
    if (m_graph is None) != (m_history is None):
        raise ValueError("Supply both the shared M graph and its history ID")
    if m_graph is not None:
        with closing(sqlite3.connect(m_graph.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            history["edges"] = retrieval_edges(db, m_history, history["turns"])
    documents = embedding_texts(history)
    texts = documents + [q["text"] for q in history["queries"]]
    fingerprint = sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest()
    if model is not None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(str(model), local_files_only=True, device="cpu")
        vectors = encoder.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        if save_vectors:
            np.savez_compressed(save_vectors, vectors=vectors, text_sha256=fingerprint)
    else:
        if vector_path is None:
            vector_path = history_path.with_suffix(".npz")
        with np.load(vector_path, allow_pickle=False) as data:
            if data["text_sha256"].item() != fingerprint:
                raise ValueError("Vector cache and input text differ")
            vectors = data["vectors"]
    if len(vectors) != len(texts) or vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid encoder output dimensions or values")
    index = Index(history, vectors[:len(documents)])
    weights = np.asarray(json.loads((HERE / "data/comparator.json").read_text())["weights"])
    return [dict(qid=q["id"], **index.retrieve(q["text"], v, weights))
            for q,v in zip(history["queries"], vectors[len(documents):], strict=True)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=EXAMPLE)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--vectors", type=Path, help="Cached embeddings matching the input texts")
    source.add_argument("--model", type=Path, help="Local all-MiniLM-L6-v2 directory")
    parser.add_argument("--save-vectors", type=Path, help="Cache embeddings from --model")
    parser.add_argument("--m-graph", type=Path, help="SQLite graph produced from the M histories")
    parser.add_argument("--m-history", help="Visibility scope in the shared M graph")
    args = parser.parse_args()
    if args.save_vectors and not args.model:
        parser.error("--save-vectors requires --model")
    if bool(args.m_graph) != bool(args.m_history):
        parser.error("--m-graph and --m-history must be used together")
    print(json.dumps(run(args.history, args.model, args.vectors, args.save_vectors,
                         m_graph=args.m_graph, m_history=args.m_history), indent=2))


if __name__ == "__main__":
    main()
