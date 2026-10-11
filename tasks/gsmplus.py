"""Helpers for the GSM-Plus tasks: the same 100 seeds as C0, perturbed.

C1 = "distraction insertion" (an irrelevant sentence with numbers; same answer as the seed).
C2 = "critical thinking" (a needed fact removed; the question has no answer).
"""
import hashlib
import json
import re
from pathlib import Path

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample_ids.json"

# qintongli/GSM-Plus test split: 8 rows per GSM8K seed, in the same type order in all 1,319
# blocks (measured on the dataset), so row = 8 * seed_index + offset.
ROWS_PER_SEED = 8
OFFSET = {"distraction insertion": 6, "critical thinking": 7}

# Last "#### ..." line of an answer: a number, or the UNANSWERABLE token allowed by prompt P1.
FINAL_LINE = re.compile(r"####\s*(UNANSWERABLE|-?[$0-9][0-9.,]*)", re.IGNORECASE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select(dataset, perturbation: str):
    items = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["items"]
    idx = [it["idx"] for it in items]
    subset = dataset.select([ROWS_PER_SEED * i + OFFSET[perturbation] for i in idx])
    for item, ptype, seed_q in zip(items, subset["perturbation_type"], subset["seed_question"], strict=True):
        if ptype != perturbation:
            raise ValueError(f"seed {item['idx']}: expected {perturbation!r}, got {ptype!r}")
        if _sha256(seed_q) != item["question_sha256"]:
            raise ValueError(f"seed {item['idx']}: GSM-Plus seed question differs from data/sample_ids.json")
    return subset.add_column("orig_idx", idx)


def process_docs_distractor(dataset):
    return _select(dataset, "distraction insertion")


def process_docs_critical(dataset):
    return _select(dataset, "critical thinking")


def classify_final_line(text: str | None) -> str:
    """'answered_number', 'abstained' (#### UNANSWERABLE) or 'no_final_answer' (no '####' followed by a number or UNANSWERABLE)."""
    found = FINAL_LINE.findall(text or "")
    if not found:
        return "no_final_answer"
    return "abstained" if found[-1].upper() == "UNANSWERABLE" else "answered_number"


def process_results_critical(doc, results):
    """Critical-thinking questions have no answer: a number on the final line is false confidence."""
    label = classify_final_line(results[0])
    return {name: int(label == name) for name in ("answered_number", "abstained", "no_final_answer")}
