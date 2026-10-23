import math

import numpy as np


def binary_scores(truth, pred):
    truth, pred = np.asarray(truth, dtype=bool), np.asarray(pred, dtype=bool)
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}


def auc(y, score):
    # Mann-Whitney with average ranks for tied scores, no sklearn dependency.
    y = np.asarray(y, dtype=bool)
    if not y.any() or y.all():
        return None
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    _, starts, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    ranks = np.empty(len(y), dtype=float)
    for start, count in zip(starts, counts):
        ranks[order[start:start + count]] = start + (count + 1) / 2
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def summarize(a):
    y = a["event"] > .5
    result = {"n": len(y), "noise_n": int((~y).sum()), "event_n": int(y.sum()),
              "detection": {str(t): binary_scores(y, a["event_prob"] >= t) for t in (.3, .5)},
              "detection_auc": auc(y, a["event_prob"])}
    for j, name in enumerate(("P", "S")):
        known = a["truth"][:, j] >= 0
        error = (a["pred"][:, j] - a["truth"][:, j]) / 100
        e = error[known]
        phase = {"annotated_n": int(known.sum()),
                 "mae_s": float(np.abs(e).mean()) if len(e) else None,
                 "rmse_s": float(np.sqrt((e * e).mean())) if len(e) else None,
                 "bias_s": float(e.mean()) if len(e) else None,
                 "unconditional_recall": {str(t): float((np.abs(e) <= t).mean()) if len(e) else None for t in (.1, .3, .5)}}
        for threshold in (.3, .5):
            # A wrong pick on an annotated trace is BOTH FP and FN. Unknown
            # phases of event traces are excluded, verified noise is included.
            emitted = (a["confidence"][:, j] >= threshold) & (a["event_prob"] >= .5)
            hit = known & emitted & (np.abs(error) <= .3)
            evaluable = known | ~y
            tp = int(hit.sum())
            fp = int((emitted & evaluable & ~hit).sum())
            fn = int((known & ~hit).sum())
            phase[f"picking_at_{threshold}"] = {"tolerance_s": .3, "tp": tp, "fp": fp, "fn": fn,
                "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "noise_false_positive_rate": float(emitted[~y].mean()) if (~y).any() else None}
        result[name] = phase
    return result


def conformal_radius(errors, alpha=.1):
    """Finite-sample split conformal: kth ORDER STATISTIC, not rounded probability."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    errors = np.asarray(errors, dtype=float)
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("Residuals must be finite and nonnegative")
    n = len(errors)
    k = math.ceil((n + 1) * (1 - alpha))
    if not n or k > n:
        return None  # No finite interval; do not fabricate a finite quantile.
    return float(np.partition(errors, k - 1)[k - 1])
