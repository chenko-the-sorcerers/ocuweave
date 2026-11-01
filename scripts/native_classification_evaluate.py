"""Recount native disease-classification predictions against frozen labels.

Accuracy target is validation >0.85; it is not a success flag for training fitting.
AUROC is one-vs-rest with average ranks for ties. No threshold is selected here.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from native_data import sha256


def auc(scores, truth):
    npos = sum(truth)
    nneg = len(truth) - npos
    if not npos or not nneg:
        return None
    ordered = sorted(zip(scores, truth), key=lambda x: x[0])
    ranks = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        ranks += ((i + 1 + j) / 2) * sum(y for _, y in ordered[i:j])
        i = j
    return (ranks - npos * (npos + 1) / 2) / (npos * nneg)


def metrics(probabilities, labels, classes):
    K = len(classes)
    if not labels or len(probabilities) != len(labels):
        raise ValueError("Empty/mismatched classification results")
    for p, y in zip(probabilities, labels):
        if (
            len(p) != K
            or not 0 <= y < K
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p)
            or not math.isclose(sum(p), 1, abs_tol=1e-5)
        ):
            raise ValueError("Invalid classification probability/label")
    predicted = [max(range(K), key=lambda c: p[c]) for p in probabilities]
    confusion = [
        [sum(y == i and h == j for y, h in zip(labels, predicted)) for j in range(K)]
        for i in range(K)
    ]
    per_class = []
    for k, name in enumerate(classes):
        tp = confusion[k][k]
        fn = sum(confusion[k]) - tp
        fp = sum(r[k] for r in confusion) - tp
        tn = len(labels) - tp - fn - fp
        per_class.append(
            {
                "class": name,
                "support": tp + fn,
                "precision": tp / (tp + fp) if tp + fp else None,
                "recall": tp / (tp + fn) if tp + fn else None,
                "specificity": tn / (tn + fp) if tn + fp else None,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                "auroc": auc(
                    [p[k] for p in probabilities], [int(y == k) for y in labels]
                ),
            }
        )

    def mean(key):
        values = [r[key] for r in per_class if r[key] is not None]
        return sum(values) / len(values) if values else None

    result = {
        "samples": len(labels),
        "accuracy": sum(y == h for y, h in zip(labels, predicted)) / len(labels),
        "balanced_accuracy": mean("recall"),
        "macro_f1": mean("f1"),
        "macro_ovr_auroc": mean("auroc"),
        "confusion_true_rows_predicted_columns": confusion,
        "per_class": per_class,
    }
    if "normal" in classes:
        normal = classes.index("normal")
        truth = [y != normal for y in labels]
        scores = [1 - p[normal] for p in probabilities]
        pred = [p >= 0.5 for p in scores]
        tp = sum(a and b for a, b in zip(truth, pred))
        fp = sum(not a and b for a, b in zip(truth, pred))
        fn = sum(a and not b for a, b in zip(truth, pred))
        tn = sum(not a and not b for a, b in zip(truth, pred))
        result["dataset_abnormality_triage"] = {
            "threshold": 0.5,
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "auroc": auc(scores, list(map(int, truth))),
            "clinical_early_detection_validated": False,
        }
    return result


def evaluate(data, run, output):
    if output.exists():
        raise FileExistsError("Preserve prior evaluation")
    meta = json.loads((data / "classification-data.json").read_text())
    native = json.loads((run / "metrics.json").read_text())
    report = {
        "classes": meta["classes"],
        "scope": "provisional image-group development; patient identity unverified",
        "test_evaluated": False,
        "targets": {"validation_classification_accuracy_gt": 0.85},
        "results": {},
    }
    for split in ("train", "validation"):
        if (
            sha256(data / (split + ".tsv"))
            != meta["partitions"][split]["labels_sha256"]
        ):
            raise ValueError("Frozen label checksum differs")
        with (data / (split + ".tsv")).open() as f:
            expected = {
                r["sample_id"]: int(r["label"])
                for r in csv.DictReader(f, delimiter="\t")
            }
        with (run / (split + "-predictions.csv")).open() as f:
            rows = list(csv.DictReader(f))
        if len({r["sample_id"] for r in rows}) != len(rows) or {
            r["sample_id"] for r in rows
        } != set(expected):
            raise ValueError("Prediction IDs differ from frozen labels")
        labels = [int(r["label"]) for r in rows]
        if any(expected[r["sample_id"]] != int(r["label"]) for r in rows):
            raise ValueError("Prediction labels changed")
        result = metrics(
            [
                [float(r["p" + str(k)]) for k in range(len(meta["classes"]))]
                for r in rows
            ],
            labels,
            meta["classes"],
        )
        if result["confusion_true_rows_predicted_columns"] != native[split][
            "confusion_true_rows_predicted_columns"
        ] or not math.isclose(
            result["accuracy"], native[split]["accuracy"], abs_tol=1e-8
        ):
            raise ValueError("Native/recount discrepancy")
        report["results"][split] = result
    report["validation_accuracy_target_met"] = (
        report["results"]["validation"]["accuracy"] > 0.85
    )
    report["clinical_validity_established"] = False
    output.mkdir(parents=True)
    (output / "classification-metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(evaluate(a.data, a.run, a.output), indent=2))
