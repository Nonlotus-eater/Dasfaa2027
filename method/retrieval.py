"""Index retained user turns, offline topic summaries and verified turn links."""
from collections import Counter, defaultdict
from functools import lru_cache
import math
import re

import numpy as np
from nltk.stem.snowball import EnglishStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .ranking import adjacency, mixed_order, nominate_indexed, rerank, turn_ranks
from .scoring import build_bm25_index, normalize_rows, score_bm25_index, score_session_vectors

STEM = EnglishStemmer()


@lru_cache(maxsize=20000)
def terms(text):
    return {STEM.stem(w) for w in re.findall(r"[a-z]+|[0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS}


def window_texts(turns):
    sessions = defaultdict(list)
    for i, t in enumerate(turns):
        sessions[t["session"]].append(i)
    windows = [""] * len(turns)
    for ids in sessions.values():
        ids.sort(key=lambda i: turns[i]["position"])
        for k, i in enumerate(ids):
            windows[i] = " ".join(turns[j]["text"] for j in ids[max(0,k-1):k+2])
    return windows


def embedding_texts(history):
    turns = history["turns"]
    return ([t["text"] for t in turns] + window_texts(turns)
            + [topic["summary"] for topic in history["topics"]])


class Index:
    def __init__(self, history, vectors):
        turns, self.topics = history["turns"], history["topics"]
        ids = {t["id"]: i for i,t in enumerate(turns)}
        n = len(turns)
        if not n or len(ids) != n or len(vectors) != 2*n + len(self.topics):
            raise ValueError("Turn identifiers must be unique and have matching embedding rows")
        self.sessions = sorted({t["session"] for t in turns})
        session_ids = {s:i for i,s in enumerate(self.sessions)}
        self.owners = np.array([session_ids[t["session"]] for t in turns])
        self.members = [np.flatnonzero(self.owners == i) for i in range(len(self.sessions))]
        self.edges = [(ids[a], ids[b]) for a,b in history["edges"]]
        if any(self.owners[a] == self.owners[b] for a,b in self.edges):
            raise ValueError("Graph edges must cross sessions")
        self.adjacent = adjacency(self.owners, self.edges)
        matrix = normalize_rows(vectors)
        self.turns, self.windows, self.summaries = matrix[:n], matrix[n:2*n], matrix[2*n:]
        self.by_session = {s:self.turns[m] for s,m in zip(self.sessions, self.members)}
        self.session_index = build_bm25_index({s:[turns[i]["text"] for i in m]
                                               for s,m in zip(self.sessions, self.members)})
        self.turn_index = build_bm25_index({str(i):[t["text"]] for i,t in enumerate(turns)})
        self.term_sets = [terms(t["text"]) for t in turns]
        self.names = set().union(*(terms(t.get("speaker", "User")) for t in turns))
        freq = Counter(w for words in self.term_sets for w in words)
        self.idf = {w: math.log((1+n)/(1+c))+1 for w,c in freq.items()}

    def features(self, text, base, dense, words, windows, mixed):
        weights = {w:self.idf.get(w, math.log(1+len(dense))+1) for w in sorted(terms(text)-self.names)}
        total = sum(weights.values()) or 1.0
        cover = np.array([sum(v for w,v in weights.items() if w in words_here)/total
                          for words_here in self.term_sets])
        mixed_rank = np.empty(len(mixed), dtype=int)
        mixed_rank[mixed] = np.arange(1, len(mixed)+1)
        joint = np.sqrt(np.maximum(dense, 0)*words)
        context = np.maximum(windows, 0)*cover
        rows = []
        for s, indices in enumerate(self.members):
            row = [base[s]]
            for channel in (dense, words, windows):
                values = channel[indices]
                row.extend((values.max(), np.sort(values)[-3:].mean()))
            row.extend((cover[indices].max(), context[indices].max(), joint[indices].max(),
                        1/math.log2(1+mixed_rank[indices].min())))
            rows.append(row)
        x = np.asarray(rows)
        span = np.ptp(x, axis=0)
        return (x-x.min(axis=0))/np.where(span > 1e-10, span, 1.0)

    def retrieve(self, text, vector, weights):
        query = normalize_rows(np.asarray(vector).reshape(1,-1))[0]
        turn = score_session_vectors(query, self.by_session, pooling="smooth_max", temperature=.05)
        topic_score = {}
        for topic, score in zip(self.topics, self.summaries @ query):
            for sid in topic["sessions"]:
                topic_score[sid] = max(topic_score.get(sid, -math.inf), float(score))
        bm25 = score_bm25_index(text, self.session_index)
        base = np.array([.75*(.90*turn[s]+.10*topic_score.get(s, turn[s]))+.25*bm25[s]
                         for s in self.sessions])
        initial = np.argsort(-base, kind="stable")
        dense = self.turns @ query
        lexical = score_bm25_index(text, self.turn_index)
        words = np.array([lexical[str(i)] for i in range(len(dense))])
        dr, wr = turn_ranks(dense, words)
        support = nominate_indexed(self.owners, dr, wr, self.adjacent, initial[:20])
        x = self.features(text, base, dense, words, self.windows @ query, mixed_order(dr, wr))
        candidates = {i for i,s in enumerate(initial) if s in support}
        order = rerank((x @ weights)[initial], candidates)
        return dict(initial=[self.sessions[i] for i in initial],
                    nominated=sorted(self.sessions[i] for i in support),
                    ranking=[self.sessions[initial[i]] for i in order])
