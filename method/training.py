"""Fit the pairwise evidence model on eligible training queries."""
import numpy as np
from sklearn.linear_model import LogisticRegression


FEATURES = ("hybrid_session", "dense_best", "dense_best3", "words_best", "words_best3",
            "context_best", "context_best3", "word_coverage", "coverage_and_context",
            "same_turn_dense_words", "best_mixed_rank")


def fit(cases, features, gold):
    differences, weights, used = [], [], []
    for case in cases:
        span = slice(case["start"], case["stop"])
        x, relevant = features[span], gold[span]
        pos, neg = x[relevant], x[~relevant]
        if not len(pos) or not len(neg):
            continue
        used.append(case["qid"])
        groups = [(neg[:6], 2/3), (neg[6:], 1/3)] if len(neg) > 6 else [(neg, 1.0)]
        for group, mass in groups:
            delta = (pos[:, None, :] - group).reshape(-1, x.shape[1])
            differences.append(delta)
            weights.append(np.full(len(delta), mass / len(delta)))
    if not used:
        raise ValueError("Training requires a query with both relevant and irrelevant candidates")
    x, w = np.concatenate(differences), np.concatenate(weights)
    model = LogisticRegression(C=1.0, fit_intercept=False, tol=1e-10, max_iter=500, solver="lbfgs")
    model.fit(np.concatenate((x, -x)), np.repeat([1, 0], len(w)), sample_weight=np.tile(w / 2, 2))
    if model.n_iter_[0] >= model.max_iter or not np.isfinite(model.coef_).all():
        raise RuntimeError("The pairwise model did not converge")
    return model.coef_[0], dict(used_queries=len(used), training_pairs=len(w))


def training_fold(population, conflicts, fold):
    held = [r for r in population if r["fold"] == fold]
    groups = {r["group"] for r in held}
    forbidden = groups.union(*(conflicts[g] for g in groups))
    train = [r for r in population if r["fold"] != fold and not r["abstention"]
             and r["group"] not in forbidden]
    if not held or not train:
        raise ValueError(f"Empty training or held-out population in fold {fold}")
    return train, held
