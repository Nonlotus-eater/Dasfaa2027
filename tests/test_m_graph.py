"""Check M-only graph construction and question-specific visibility."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from build_m_graph import apply_decisions, build
from method.m_graph import neighbors, propose, retrieval_edges
from method.ranking import adjacency
from retrieve import EXAMPLE, run


OLD = "I repaired the greenhouse humidity controller and tested the sensor on Monday."
NEW = "I replaced the greenhouse humidity controller and calibrated the sensor on Friday."


def history(qid, sessions):
    return {"question_id": qid, "question": "unused", "answer_session_ids": ["unused"],
            "haystack_session_ids": [sid for sid, _ in sessions],
            "haystack_dates": [f"2023/01/{i+1:02d}" for i in range(len(sessions))],
            "haystack_sessions": [[{"role": "user", "content": t, "has_answer": True}] for _, t in sessions]}


class SharedGraphTests(unittest.TestCase):
    def test_one_graph_distinguishes_order_visibility_and_content(self):
        rows = [history("both", [("a", OLD), ("b", NEW)]), history("one", [("a", OLD)]),
                history("also_both", [("a", OLD), ("z", "Another session."), ("b", NEW)]),
                history("reverse", [("b", NEW), ("a", OLD)]),
                history("changed", [("a", "I read the weather forecast."), ("b", NEW)])]
        with tempfile.TemporaryDirectory() as directory:
            data, path = Path(directory) / "histories.json", Path(directory) / "m.sqlite3"
            data.write_text(json.dumps(rows))
            self.assertEqual(build(data, path)["proposed"], 1)
            with self.assertRaises(FileExistsError):
                build(data, path)
            with closing(sqlite3.connect(path)) as db, db:
                key, = db.execute("SELECT key FROM candidates").fetchone()
                proof = dict(candidate=key, origin="M-shared", model="gpt-4o-mini-2024-07-18",
                             request_id="fixture", cosine=.8, relation="same_matter",
                             old=[0,len(OLD),OLD], new=[0,len(NEW),NEW])
                for bad in (dict(origin="external"), dict(candidate="other"), dict(new=[0,3,"xyz"])):
                    with self.assertRaises(ValueError):
                        apply_decisions(db, [dict(proof, **bad)])
                self.assertEqual(db.execute("SELECT count(*) FROM edges").fetchone()[0], 0)
                apply_decisions(db, [proof, proof])
                with self.assertRaises(ValueError):
                    apply_decisions(db, [dict(proof, relation="activity_history")])
                self.assertEqual(db.execute("SELECT count(*) FROM edges").fetchone()[0], 1)
                self.assertEqual(len(propose(db)["pairs"]), 1)
                self.assertEqual(db.execute("SELECT count(*) FROM proofs").fetchone()[0], 1)
                turns = [dict(session="a", id="a0", text=OLD), dict(session="b", id="b0", text=NEW)]
                self.assertEqual(retrieval_edges(db, "both", turns), [("a0","b0")])
                self.assertEqual(retrieval_edges(db, "also_both", turns), [("a0","b0")])
                self.assertEqual(neighbors(db, "both", ["b"])[0]["session"], "a")
                for qid in ("one", "reverse", "changed"):
                    self.assertEqual(neighbors(db, qid, ["a"]), [])
                    self.assertEqual(retrieval_edges(db, qid, turns), [])
                with self.assertRaises(ValueError):
                    retrieval_edges(db, "absent", turns)
                db.execute("UPDATE metadata SET value='null' WHERE key='build_policy'")
                with self.assertRaises(ValueError):
                    propose(db)

    def test_retrieve_uses_one_projection_for_multiple_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw, graph = folder / "m.json", folder / "m.sqlite3"
            raw.write_text(json.dumps([history("visible", [("a", OLD)])]))
            build(raw, graph)
            example = json.loads(EXAMPLE.read_bytes())
            example["edges"] = []
            reference = folder / "example.json"
            reference.write_text(json.dumps(example))
            expected = run(reference, vector_path=EXAMPLE.with_suffix(".npz"))
            with patch("method.retrieval.adjacency", wraps=adjacency) as create:
                actual = run(EXAMPLE, m_graph=graph, m_history="visible")
            self.assertEqual(actual, expected)
            self.assertGreater(len(actual), 1)
            self.assertEqual(create.call_count, 1)
