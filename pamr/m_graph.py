"""Build one LongMemEval-M association graph and project it onto visible histories."""
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
import json
import math
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

BUILD_POLICY = "m-only-shared-v1"
SHORTLIST = 768
JUDGE_LIMIT = 256
MAX_WORD_FREQUENCY = 64
WORDS = re.compile(r"[a-z]{4,}", re.I | re.ASCII)
PERSONAL = re.compile(r"\b(?:I|my|we|our)\b", re.I)


DDL = """
CREATE TABLE questions(qid TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE sessions(id INTEGER PRIMARY KEY, source_id TEXT NOT NULL, content_sha256 BLOB NOT NULL,
                      UNIQUE(source_id,content_sha256));
CREATE TABLE turns(id INTEGER PRIMARY KEY, session INTEGER NOT NULL REFERENCES sessions, pos INTEGER NOT NULL,
                   role TEXT NOT NULL, text TEXT NOT NULL, UNIQUE(session,pos));
CREATE TABLE visibility(qid TEXT NOT NULL REFERENCES questions, position INTEGER NOT NULL,
                        session INTEGER NOT NULL REFERENCES sessions, date TEXT NOT NULL,
                        PRIMARY KEY(qid,position)) WITHOUT ROWID;
CREATE INDEX visible_session ON visibility(session,qid,position);
CREATE INDEX visible_question ON visibility(qid,session,position);
CREATE TABLE edges(id INTEGER PRIMARY KEY, old_turn INTEGER NOT NULL REFERENCES turns,
                   new_turn INTEGER NOT NULL REFERENCES turns, kind TEXT NOT NULL,
                   old_start INTEGER NOT NULL,old_end INTEGER NOT NULL,old_quote TEXT NOT NULL,
                   new_start INTEGER NOT NULL,new_end INTEGER NOT NULL,new_quote TEXT NOT NULL,
                   UNIQUE(old_turn,new_turn,kind,old_start,old_end,new_start,new_end));
CREATE INDEX edges_new ON edges(new_turn);
CREATE TABLE proofs(edge INTEGER NOT NULL REFERENCES edges, origin TEXT NOT NULL CHECK(origin='M-shared'), batch TEXT NOT NULL,
                   pair TEXT NOT NULL, artifact_sha256 TEXT NOT NULL, PRIMARY KEY(origin,batch,pair)) WITHOUT ROWID;
CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
"""


def pack(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def content_hash(turns):
    return sha256(pack(turns)).digest()


def meta(db, key, value):
    db.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)", (key, json.dumps(value, ensure_ascii=False)))


def require_m_only(db):
    policy = db.execute("SELECT value FROM metadata WHERE key='build_policy'").fetchone()
    if policy != (json.dumps(BUILD_POLICY),):
        raise ValueError("This database was not built by the M-only shared-graph pipeline")
    if db.execute("SELECT 1 FROM proofs WHERE origin != 'M-shared' LIMIT 1").fetchone():
        raise ValueError("Unexpected proof origin in the M graph")


def index_histories(db, records):
    """Each record is a history ID, its source fingerprint and its M source fields."""
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript(DDL)
    sessions = {}
    stats = Counter()
    for qid, checksum, source in records:
        ingest_history(db, qid, source, checksum, sessions, stats)
        stats["histories"] += 1
    if not stats["histories"]:
        raise ValueError("No M histories were provided")
    meta(db, "build_policy", BUILD_POLICY)
    meta(db, "source_counts", stats)
    return dict(stats)


def ingest_history(db, qid, src, checksum, sessions, stats):
    """Index only source sessions. Session content is part of its identity."""
    rows = zip(src["haystack_dates"], src["haystack_session_ids"], src["haystack_sessions"], strict=True)
    # Official loader uses stable chronological order, including ties.
    ordered = sorted(rows, key=lambda item: item[0])
    db.execute("INSERT INTO questions VALUES(?,?)", (qid, checksum))
    for position, (date, sid, messages) in enumerate(ordered):
        turns = [(i, t["role"], t["content"]) for i, t in enumerate(messages) if t["content"].strip()]
        key = (sid, content_hash(turns))
        session = sessions.get(key)
        if session is None:
            session = db.execute("INSERT INTO sessions(source_id,content_sha256) VALUES(?,?)", key).lastrowid
            sessions[key] = session
            db.executemany("INSERT INTO turns(session,pos,role,text) VALUES(?,?,?,?)",
                           ((session, *turn) for turn in turns))
        db.execute("INSERT INTO visibility VALUES(?,?,?,?)", (qid, position, session, date))
        stats["session_occurrences"] += 1
        stats["nonempty_turn_occurrences"] += len(turns)


def co_visible(order, older, newer):
    return any(qid in order[newer] and first < order[newer][qid][1]
               for qid, (first, _) in order[older].items())


def corpus_digest(db):
    return sha256(pack(list(db.execute("SELECT qid,source_sha256 FROM questions ORDER BY qid")))).hexdigest()


def propose(db):
    """Rank candidates over the entire M corpus with at least one chronological witness."""
    require_m_only(db)
    order = defaultdict(dict)
    for sid, qid, first, last in db.execute(
            "SELECT session,qid,min(position),max(position) FROM visibility GROUP BY session,qid"):
        order[sid][qid] = first, last
    names = dict(db.execute("SELECT id,source_id FROM sessions"))
    docs, words, postings = {}, {}, defaultdict(list)
    for node, session, txt in db.execute("SELECT id,session,text FROM turns WHERE role='user' ORDER BY id"):
        if len(txt) < 40 or len(("User: " + txt).encode()) > 2048 or not PERSONAL.search(txt):
            continue
        claims = [s for s in re.split(r'(?<=[.!?])\s+|\n+', txt) if '?' not in s and PERSONAL.search(s)]
        tokens = set(WORDS.findall(" ".join(claims).lower())) - ENGLISH_STOP_WORDS
        if len(tokens) < 4:
            continue
        docs[node] = session
        words[node] = tokens
        for t in tokens:
            postings[t].append(node)
    shared = Counter()
    for nodes in postings.values():
        if 2 <= len(nodes) <= MAX_WORD_FREQUENCY:
            shared.update(combinations(nodes, 2))
    proposals = []
    for (a, b), n in shared.items():
        sa, sb = docs[a], docs[b]
        if n < 2 or names[sa] == names[sb]:
            continue
        reverse = not co_visible(order, sa, sb)
        if reverse and not co_visible(order, sb, sa):
            continue
        overlap = words[a] & words[b]
        score = sum(math.log((len(docs) + 1) / len(postings[t])) for t in sorted(overlap))
        score /= math.sqrt(len(words[a]) * len(words[b]))
        if reverse:
            a, b, sa, sb = b, a, sb, sa
        proposals.append((score, a, b, sa, sb))
    used, chosen = set(), []
    for score, a, b, sa, sb in sorted(proposals, key=lambda p: (-p[0], p[1], p[2])):
        if sa in used or sb in used:
            continue
        witness = next(q for q, (first, _) in sorted(order[sa].items())
                       if q in order[sb] and first < order[sb][q][1])
        chosen.append(dict(old=a, new=b, lexical=score, witness=witness,
                           positions=[order[sa][witness][0], order[sb][witness][1]]))
        used.update([sa, sb])
        if len(chosen) >= SHORTLIST:
            break
    return dict(build_policy=BUILD_POLICY, corpus_sha256=corpus_digest(db), eligible_user_turns=len(docs),
                co_visible_lexical_pairs=len(proposals),
                  policy=dict(rare_word_document_frequency=[2, MAX_WORD_FREQUENCY], minimum_distinct_rare_words=2,
                              unique_sessions_per_shortlist=True, max_shortlist=SHORTLIST,
                              max_judge_pairs=JUDGE_LIMIT), pairs=chosen)


def resolve_span(db, session, msg, span):
    pos = int(msg["source_id"].removeprefix(msg["session_id"] + ":message:"))
    node, role, raw = db.execute("SELECT id,role,text FROM turns WHERE session=? AND pos=?",
                                 (session, pos)).fetchone()
    prefix = msg["speaker"] + ": "
    base = msg["original_char_start"]
    assert role == msg["role"] and msg["text"] == prefix + raw[base:msg["original_char_end"]]
    a, b, quote = span["start"], span["end"], span["text"]
    assert 0 <= a < b <= len(msg["text"]) and msg["text"][a:b] == quote
    lo, hi = base + max(a, len(prefix)) - len(prefix), base + b - len(prefix)
    if hi <= lo:
        return None
    quote = quote[max(0, len(prefix) - a):]
    assert 0 <= lo < hi <= len(raw) and quote == raw[lo:hi]
    return node, lo, hi, quote


def add_edge(db, old, new, kind, origin, batch, pair_id, checksum):
    if origin != "M-shared" or kind not in {"same_matter", "activity_history"}:
        raise ValueError("Unexpected M verification origin or relation")
    for node, begin, end, quote in (old, new):
        row = db.execute("SELECT text FROM turns WHERE id=?", (node,)).fetchone()
        if not row or not 0 <= begin < end <= len(row[0]) or row[0][begin:end] != quote:
            raise ValueError("Evidence does not match the M source turn")
    a, lo, hi, old_quote = old
    b, start, end, new_quote = new
    key = a, b, kind, lo, hi, start, end
    previous = db.execute("""SELECT e.old_turn,e.new_turn,e.kind,e.old_start,e.old_end,e.new_start,e.new_end,
                             p.artifact_sha256 FROM proofs p JOIN edges e ON e.id=p.edge
                             WHERE p.origin=? AND p.batch=? AND p.pair=?""", (origin, batch, pair_id)).fetchone()
    if previous is not None:
        if previous != (*key, checksum):
            raise ValueError("The saved M proof conflicts with this verification")
        return
    db.execute("""INSERT OR IGNORE INTO edges(old_turn,new_turn,kind,old_start,old_end,old_quote,
                  new_start,new_end,new_quote) VALUES(?,?,?,?,?,?,?,?,?)""",
               (a, b, kind, lo, hi, old_quote, start, end, new_quote))
    edge = db.execute("""SELECT id FROM edges WHERE old_turn=? AND new_turn=? AND kind=? AND old_start=?
                      AND old_end=? AND new_start=? AND new_end=?""", key).fetchone()[0]
    db.execute("INSERT INTO proofs VALUES(?,?,?,?,?)", (edge, origin, batch, pair_id, checksum))


def retrieval_edges(db, qid, turns):
    """Map chronologically visible graph endpoints to retained turn IDs in prepared memory."""
    require_m_only(db)
    if not db.execute("SELECT 1 FROM questions WHERE qid=?", (qid,)).fetchone():
        raise ValueError("Unknown M history ID")
    keys = defaultdict(list)
    for turn in turns:
        keys[turn["session"], turn["text"]].append(turn)

    def align(session, text, pos):
        matches = [t["id"] for t in keys.get((session, text), [])
                   if t.get("source_position", pos) == pos]
        return matches[0] if len(matches) == 1 else None

    rows = db.execute("""WITH scoped AS (
        SELECT session,min(position) first_pos,max(position) last_pos
        FROM visibility WHERE qid=? GROUP BY session)
        SELECT sa.source_id,x.text,x.pos,sb.source_id,y.text,y.pos
        FROM edges e JOIN turns x ON x.id=e.old_turn JOIN turns y ON y.id=e.new_turn
        JOIN scoped a ON a.session=x.session JOIN scoped b ON b.session=y.session
        JOIN sessions sa ON sa.id=x.session JOIN sessions sb ON sb.id=y.session
        WHERE a.first_pos < b.last_pos ORDER BY e.id""", (qid,))
    aligned = set()
    for a, text_a, pos_a, b, text_b, pos_b in rows:
        old, new = align(a, text_a, pos_a), align(b, text_b, pos_b)
        if old is not None and new is not None:
            aligned.add((old, new))
    return sorted(aligned)


def neighbors(db, qid, seeds, limit=5):
    if not db.execute("SELECT 1 FROM questions WHERE qid=?", (qid,)).fetchone():
        raise ValueError("Unknown M question ID")
    if not seeds:
        return []
    # Both endpoints must be visible, in the evidence's original chronological direction.
    # UNION makes the same edge discoverable from either side of a seed.
    seed_values = ",".join("(?)" for _ in seeds)
    rows = db.execute(f"""
      WITH seed_ids(sid) AS (VALUES {seed_values}), scoped AS MATERIALIZED (
        SELECT v.session, min(v.position) first_pos,max(v.position) last_pos,s.source_id sid
        FROM visibility v JOIN sessions s ON s.id=v.session WHERE v.qid=? GROUP BY v.session
      ), seed_turns AS MATERIALIZED (
        SELECT t.id FROM seed_ids i JOIN scoped s ON s.sid=i.sid JOIN turns t ON t.session=s.session
      ), candidates(eid) AS (
        SELECT e.id FROM seed_turns t JOIN edges e ON e.old_turn=t.id UNION
        SELECT e.id FROM seed_turns t JOIN edges e ON e.new_turn=t.id
      )
      SELECT e.id, a.sid, b.sid, e.kind, e.old_quote, e.new_quote
      FROM candidates c JOIN edges e ON e.id=c.eid
      JOIN turns ta ON ta.id=e.old_turn JOIN scoped a ON a.session=ta.session
      JOIN turns tb ON tb.id=e.new_turn JOIN scoped b ON b.session=tb.session
      WHERE a.first_pos < b.last_pos AND (a.sid IN seed_ids) != (b.sid IN seed_ids)
      ORDER BY e.id""", (*seeds, qid))
    seed_set, by_target = set(seeds), defaultdict(list)
    for eid, old, new, kind, old_quote, new_quote in rows:
        target = new if old in seed_set else old
        by_target[target].append(dict(edge=eid, old_session=old, new_session=new, kind=kind,
                                      old_quote=old_quote, new_quote=new_quote))
    return [dict(session=t, edge_count=len(proofs), example=proofs[0]) for t, proofs in
            sorted(by_target.items(), key=lambda item: (-len(item[1]), item[0]))[:limit]]


