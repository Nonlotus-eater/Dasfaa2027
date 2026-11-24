"""Train and evaluate the retrieval method on the matched LoCoMo and LongMemEval paper cohorts."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from threadpoolctl import threadpool_limits

from method.ranking import nominate, rerank
from method.training import FEATURES, fit, training_fold

DATA = Path(__file__).resolve().parent / "data"
METRICS = ("Success@1", "Success@5", "Success@10", "Recall@1", "Recall@5", "Recall@10", "MRR")
DATASETS = ("LoCoMo", "S", "M")


class Dataset:
    def __init__(self, directory, name):
        self.name = name
        manifest = json.loads((directory / f"{name}.json").read_text())
        self.cases = manifest["cases"]
        self.conflicts = {int(g): ns for g, ns in manifest["conflicts"].items()}
        self.evaluation = [r for r in self.cases if r["evaluation"]]
        if len({r["qid"] for r in self.cases}) != len(self.cases):
            raise ValueError("Duplicate benchmark query")
        parts = defaultdict(list)
        for filename in manifest["parts"]:
            with np.load(directory / filename, allow_pickle=False) as data:
                for key in ("features", "gold", "legacy"):
                    parts[key].append(data[key])
        self.features, self.gold, self.legacy = (np.concatenate(parts[k]) for k in ("features", "gold", "legacy"))
        with gzip.open(directory / f"{name}-graph.json.gz", "rt") as f:
            self.graphs = json.load(f)
        if set(self.graphs) != {r["qid"] for r in self.evaluation}:
            raise ValueError("Graph and evaluation query IDs differ")
        if self.features.shape != (len(self.gold), len(FEATURES)) or self.legacy.shape != self.gold.shape:
            raise ValueError("Feature and session counts differ")
        end = 0
        for r in self.cases:
            if r["start"] != end or r["stop"] <= end or r["gold_count"] < 1:
                raise ValueError("Invalid candidate span or gold count")
            end = r["stop"]
        if end != len(self.gold):
            raise ValueError("Candidate spans do not cover the feature matrix")
        self.support = {}
        for r in self.evaluation:
            q = r["qid"]
            nodes, edges = self.graphs[q]
            owners, dense, words = np.asarray(nodes, dtype=int).reshape(-1, 3).T
            n = r["stop"] - r["start"]
            self.support[q] = nominate(owners, dense, words, edges, range(min(20, n)))

    def predict(self, cases, weights, mode):
        for r in cases:
            span = slice(r["start"], r["stop"])
            scores = self.features[span] @ weights
            old = set(np.flatnonzero(self.legacy[span]))
            graph = self.support[r["qid"]]
            if not set(graph) <= old:
                raise ValueError("A turn-level nomination is missing from the session-graph control")
            row = dict(dataset=self.name, qid=r["qid"], model=mode, fold=r["fold"],
                       nominations_old=len(old), nominations_new=len(graph))
            for rule, support in (("old", old), ("new", graph)):
                order = rerank(scores, support)
                hits = np.flatnonzero(self.gold[span][order]) + 1
                row[rule+"_first"] = int(hits[0]) if len(hits) else None
                row[rule+"_last"] = int(hits[-1]) if len(hits) == r["gold_count"] else None
            if mode in ("transfer", "holdout"):
                if [row["old_first"], row["old_last"]] != r["legacy_ranks"]:
                    raise ValueError(f"Original main-table control differs for {self.name}/{r['qid']}")
            yield row


def percentages(rows, rule):
    totals = np.zeros(7)
    for r in rows:
        first, last = r[rule+"_first"], r[rule+"_last"]
        totals[:6] += [value is not None and value <= k for value in (first,last) for k in (1,5,10)]
        totals[6] += 1/first if first else 0
    return dict(zip(METRICS, 100*totals/len(rows)))


def evaluate(directory, mode="both", population="paper"):
    locomo = Dataset(directory, "LoCoMo")
    dev = [r for r in locomo.cases if not r["evaluation"]]
    if {r["group"] for r in dev} & {r["group"] for r in locomo.evaluation}:
        raise ValueError("LoCoMo train and test share a conversation")
    weights, counts = fit(dev, locomo.features, locomo.gold)
    frozen = json.loads((directory / "comparator.json").read_text())
    if list(FEATURES) != frozen["columns"] or counts["training_pairs"] != frozen["training_pairs"]:
        raise ValueError("The LoCoMo feature schema or training population differs")
    np.testing.assert_allclose(weights, frozen["weights"], atol=1e-6, rtol=1e-6)
    models = [dict(dataset="LoCoMo", model="holdout", development=len(dev), held=len(locomo.evaluation),
                   weights=weights.tolist(), **counts)]
    results = list(locomo.predict(locomo.evaluation, weights, "holdout"))
    for ds in ("S", "M"):
        data = Dataset(directory, ds)
        if mode in ("transfer", "both"):
            results.extend(data.predict(data.evaluation, weights, "transfer"))
        if mode in ("fivefold", "both"):
            cohort = data.evaluation if population == "paper" else data.cases
            predicted = set()
            for fold in range(1, 6):
                train, held = training_fold(cohort, data.conflicts, fold)
                fold_weights, counts = fit(train, data.features, data.gold)
                test = [r for r in held if r["evaluation"]]
                results.extend(data.predict(test, fold_weights, "fivefold"))
                ids = {r["qid"] for r in test}
                if predicted & ids:
                    raise ValueError("Repeated held-out predictions")
                predicted.update(ids)
                models.append(dict(dataset=ds, model="fivefold", fold=fold, eligible_train=len(train),
                                   held=len(held), reported=len(test), weights=fold_weights.tolist(), **counts))
            if predicted != set(data.graphs):
                raise ValueError(f"Incomplete held-out predictions for {ds}")
    groups = defaultdict(list)
    for r in results:
        groups[r["dataset"],r["model"]].append(r)
    panels = []
    for (dataset, protocol), rows in groups.items():
        old, new = percentages(rows, "old"), percentages(rows, "new")
        panels.append(dict(dataset=dataset, model=protocol, n=len(rows), old=old, corrected=new,
                           graph_delta_pp={k:new[k]-old[k] for k in METRICS}))
    return dict(fold_population=population, models=models, panels=panels,
                m_graph_source="Historical shared graph with M judgments and projected S judgments"), results


def write_outputs(output, summary, rows):
    output.mkdir(parents=True, exist_ok=True)
    for name in ("metrics.json", "per-query.csv"):
        with NamedTemporaryFile("w", encoding="utf-8", dir=output, newline="", delete=False) as f:
            if name.endswith(".json"):
                json.dump(summary, f, indent=2, allow_nan=False)
                f.write("\n")
            else:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        Path(f.name).replace(output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("transfer", "fivefold", "both"), default="both")
    parser.add_argument("--fold-population", choices=("paper","archive"), default="paper",
                        help="Train within the 400 paper queries or the historical 500-query folds")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, help="Write metrics.json and per-query.csv")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summary, rows = evaluate(args.data, args.model, args.fold_population)
    print("Dataset   Model       N     " + "  ".join(METRICS))
    for p in summary["panels"]:
        print(f"{p['dataset']:8}  {p['model']:8}  {p['n']:4}   " +
              "  ".join(f"{p['corrected'][k]:7.2f}" for k in METRICS))
    if args.output:
        write_outputs(args.output, summary, rows)


if __name__ == "__main__":
    main()
