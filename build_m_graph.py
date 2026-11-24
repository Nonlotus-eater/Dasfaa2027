"""Build and inspect a single shared graph from LongMemEval-M source histories."""
import argparse
from contextlib import closing
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sqlite3
from tempfile import NamedTemporaryFile

from pamr.m_graph import add_edge, index_histories, meta, neighbors, pack, propose, require_m_only


def array_items(path):
    """Stream the official JSON array without keeping the full corpus in memory."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as source:
        buf = ""

        def refill():
            nonlocal buf
            chunk = source.read(1 << 20)
            buf += chunk
            return bool(chunk)

        def skip_space():
            nonlocal buf
            buf = buf.lstrip()
            while not buf and refill():
                buf = buf.lstrip()

        skip_space()
        if not buf.startswith("["):
            raise ValueError("Expected a JSON array of M histories")
        buf = buf[1:]
        skip_space()
        if not buf.startswith("]"):
            while True:
                try:
                    row, end = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    if refill():
                        continue
                    raise
                yield row
                buf = buf[end:]
                skip_space()
                if buf.startswith("]"):
                    break
                if not buf.startswith(","):
                    raise ValueError("Expected a comma between M histories")
                buf = buf[1:]
                skip_space()
        buf = buf[1:]
        skip_space()
        if buf:
            raise ValueError("Unexpected text after the history array")


def source_only(row):
    return dict(haystack_dates=row["haystack_dates"], haystack_session_ids=row["haystack_session_ids"],
                haystack_sessions=[[{"role": m["role"], "content": m["content"]} for m in messages]
                                   for messages in row["haystack_sessions"]])


def source_records(data):
    if data.is_dir():
        for path in sorted(data.glob("*.json")):
            src = source_only(json.loads(path.read_bytes()))
            yield path.stem, sha256(pack(src)).hexdigest(), src
        return
    # Stable history order gives the same turn IDs for the official array and split files.
    with closing(sqlite3.connect("")) as staging:
        staging.execute("CREATE TABLE histories(qid TEXT PRIMARY KEY, source BLOB) WITHOUT ROWID")
        staging.executemany("INSERT INTO histories VALUES(?,?)",
                            ((r["question_id"], pack(source_only(r))) for r in array_items(data)))
        for qid, raw in staging.execute("SELECT qid,source FROM histories ORDER BY qid"):
            yield qid, sha256(raw).hexdigest(), json.loads(raw)


def endpoint(db, node):
    row = db.execute("""SELECT s.source_id,hex(s.content_sha256),t.pos FROM turns t
                        JOIN sessions s ON s.id=t.session WHERE t.id=?""", (node,)).fetchone()
    return [row[0], row[1].lower(), row[2]]


def candidate_key(db, pair):
    return sha256(pack([endpoint(db, pair[side]) for side in ("old", "new")])).hexdigest()


def build(data, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Choose a new output filename: {target}")
    with NamedTemporaryFile(dir=target.parent, suffix=".building", delete=False) as file:
        temp = Path(file.name)
    try:
        with closing(sqlite3.connect(temp)) as db, db:
            counts = index_histories(db, source_records(data))
            plan = propose(db)
            db.execute("""CREATE TABLE candidates(key TEXT PRIMARY KEY, old_turn INTEGER REFERENCES turns,
                       new_turn INTEGER REFERENCES turns, witness TEXT REFERENCES questions, lexical REAL)""")
            db.executemany("INSERT INTO candidates VALUES(?,?,?,?,?)",
                           ((candidate_key(db, p), p["old"], p["new"], p["witness"], p["lexical"])
                            for p in plan["pairs"]))
            meta(db, "proposal_policy", plan["policy"])
            counts["proposed"] = len(plan["pairs"])
        os.link(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    return counts


def apply_decisions(db, rows):
    """Import upstream positive M judgments only for candidates of this source index."""
    require_m_only(db)
    applied = 0
    for row in rows:
        if (row["origin"] != "M-shared" or row["model"] != "gpt-4o-mini-2024-07-18"
                or not row["request_id"] or not math.isfinite(row["cosine"]) or not .40 <= row["cosine"] <= 1.000001):
            raise ValueError("Expected a verified M source judgment passing the cosine threshold")
        pair = db.execute("SELECT old_turn,new_turn FROM candidates WHERE key=?", (row["candidate"],)).fetchone()
        if pair is None:
            raise ValueError("The judged source pair is absent from the M shortlist")
        old, new = [(node, *row[side]) for node, side in zip(pair, ("old", "new"), strict=True)]
        checksum = sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        add_edge(db, old, new, row["relation"], row["origin"], row["request_id"], row["candidate"], checksum)
        applied += 1
    return applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="task", required=True)
    create = sub.add_parser("propose", help="Create one SQLite graph and a corpus-wide M shortlist")
    create.add_argument("data", type=Path, help="Official M JSON array or directory of M history JSON files")
    create.add_argument("database", type=Path)
    apply = sub.add_parser("apply", help="Attach validated M judgments from a JSONL file")
    apply.add_argument("database", type=Path)
    apply.add_argument("judgments", type=Path)
    lookup = sub.add_parser("neighbors", help="Project verified edges onto one history")
    lookup.add_argument("database", type=Path)
    lookup.add_argument("qid")
    lookup.add_argument("seeds", nargs="+")
    args = parser.parse_args()
    if args.task == "propose":
        result = build(args.data, args.database)
    else:
        mode = "rw" if args.task == "apply" else "ro"
        with closing(sqlite3.connect(args.database.resolve().as_uri() + "?mode=" + mode, uri=True)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            require_m_only(db)
            if args.task == "apply":
                with args.judgments.open() as source:
                    result = {"processed": apply_decisions(db, map(json.loads, source))}
            else:
                result = neighbors(db, args.qid, args.seeds)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
