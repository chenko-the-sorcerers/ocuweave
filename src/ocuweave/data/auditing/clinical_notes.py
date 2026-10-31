"""Deterministic review flags, not a clinical negation or diagnosis classifier.

Only aggregate counts and filename-based review flags may leave the input table.
No note text, generated summary, or demographic value is written to the report.
"""
from collections import Counter, defaultdict
import hashlib
import re


PATTERNS = {
    "diagnosis_term": r"\b(?:glaucoma\w*|poag|ntg|pacg|oag|coag)\b",
    "suspect_context": r"\b(?:suspect\w*|rule\s+out|r/o|risk)\b",
    "negation_context": r"\b(?:no|not|without|denies|negative|unlikely)\b",
    "family_context": r"\b(?:family|fhx|f/h|mother|father|sister|brother)\b",
    "treatment_proxy": r"\b(?:latanoprost|timolol|dorzolamide|brimonidine|cosopt|trabeculectomy|trabeculoplasty|slt)\b",
    "endpoint_proxy": r"\b(?:visual\s+field|hvf|rnfl|mean\s+deviation|optic\s+nerve)\b",
}


def note_flags(text):
    """Broad co-occurrence only: a negation flag does not negate a diagnosis."""
    normalized = " ".join(text.casefold().split())
    return {key: bool(re.search(pattern, normalized)) for key, pattern in PATTERNS.items()}


def audit_notes(rows):
    counts = defaultdict(Counter)
    groups = defaultdict(list)
    review = []
    for row in rows:
        # Study leakage on development data; leave the test note contents sealed.
        if row["use"] not in {"training", "validation"}:
            continue
        text = row["note"].strip()
        flags = note_flags(text)
        split = row["use"]
        counts[split]["rows"] += 1
        counts[split]["missing_note"] += not bool(text)
        counts[split].update({key: int(value) for key, value in flags.items()})
        review.append({"filename": row["filename"], "split": split, **flags})
        if text:
            digest = hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()
            groups[digest].append({"filename": row["filename"], "split": split})
    duplicates = [items for items in groups.values() if len(items) > 1]
    return {
        "protocol": "fairvlmed-note-review-v1",
        "splits_read": ["training", "validation"],
        "counts": dict(counts),
        "patterns": PATTERNS,
        "normalized_duplicate_groups": duplicates,
        "cross_development_split_duplicate_groups": sum(
            len({r["split"] for r in group}) > 1 for group in duplicates
        ),
        "review_queue": review,
        "raw_notes_logged": False,
        "semantic_review_complete": False,
        "limitations": [
            "Flags are lexical co-occurrence, not scope-aware negation or confirmed leakage.",
            "Absence of a flag does not establish target-free text.",
            "Generated summaries are excluded. No redacted note is claimed safe for training.",
            "Repeated notes may be templates; no patient identities are inferred.",
        ],
    }
