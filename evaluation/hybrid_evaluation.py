"""Validation-only calibration and held-out evaluation helpers."""
import hashlib, json
from pathlib import Path
from metrics import confusion_counts, precision_recall_f1

def validate_disjoint_splits(validation_rows, test_rows):
    for field in ("topic", "template"):
        left = {row[field] for row in validation_rows}
        right = {row[field] for row in test_rows}
        overlap = left & right
        if overlap:
            raise ValueError(f"validation/test {field} leakage: {sorted(overlap)}")

def calibrate(rows, weight_grid=None, thresholds=None):
    """Maximize validation F1, then specificity, then deterministic config order."""
    weight_grid = weight_grid or [
        {"semantic": .2, "lexical": .4, "phrase": .4},
        {"semantic": .4, "lexical": .3, "phrase": .3},
        {"semantic": .6, "lexical": .2, "phrase": .2},
        {"semantic": .8, "lexical": .1, "phrase": .1},
    ]
    thresholds = thresholds or [i / 100 for i in range(10, 91, 5)]
    if not rows: raise ValueError("validation data cannot be empty")
    best = None
    for weights in weight_grid:
        for threshold in thresholds:
            y = [int(r["relevant"]) for r in rows]
            scores = [sum(weights[k] * float(r[k]) for k in weights) / sum(weights.values()) for r in rows]
            counts = confusion_counts(y, [int(s >= threshold) for s in scores])
            precision, recall, f1 = precision_recall_f1(counts)
            specificity = counts["tn"] / (counts["tn"] + counts["fp"]) if counts["tn"] + counts["fp"] else 0
            key = (f1, specificity, -threshold, tuple(weights.values()))
            if best is None or key > best[0]: best = (key, weights.copy(), threshold, counts, precision, recall, f1, specificity)
    _, weights, threshold, counts, precision, recall, f1, specificity = best
    return {"weights": weights, "threshold": threshold, "validation_metrics": {**counts, "precision": precision, "recall": recall, "f1": f1, "specificity": specificity}}

def evaluate(rows, config):
    weights, threshold = config["weights"], float(config["threshold"])
    y = [int(r["relevant"]) for r in rows]
    scores = [sum(weights[k] * float(r[k]) for k in weights) / sum(weights.values()) for r in rows]
    counts = confusion_counts(y, [int(s >= threshold) for s in scores])
    p, r, f = precision_recall_f1(counts)
    tn, fp = counts["tn"], counts["fp"]
    return {**counts, "precision": p, "recall": r, "f1": f, "specificity": tn / (tn + fp) if tn + fp else 0, "false_positive_rate": fp / (fp + tn) if fp + tn else 0, "scores": scores}

def freeze_config(path: Path, config, validation_path: Path):
    payload = {**config, "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(), "calibration_source": "validation_only", "frozen": True}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
