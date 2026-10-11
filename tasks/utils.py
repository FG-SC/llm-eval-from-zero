"""Helpers referenced by the task YAMLs in this folder (loaded by lm_eval through !function)."""
import hashlib
import json
from pathlib import Path

SAMPLE_FILE = Path(__file__).resolve().parents[1] / "data" / "sample_ids.json"

# P0: the same zero-shot prompt for every condition.
P0_TEMPLATE = (
    "Solve the following math problem. Think step by step.\n"
    "On the last line, write the final answer as: #### <number>\n\n"
    "Problem: {question}"
)
# P1: P0 plus explicit permission not to answer (the missing-fact runs and their control on the original
# questions).
P1_TEMPLATE = (
    "Solve the following math problem. Think step by step.\n"
    "On the last line, write the final answer as: #### <number>\n"
    "If the problem does not give enough information to compute a single answer, "
    "write on the last line: #### UNANSWERABLE\n\n"
    "Problem: {question}"
)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_docs_seed100(dataset):
    """Keep only the 100 sampled questions, in ascending GSM8K index order.

    Uses select() instead of lm_eval's --samples flag (which mislabels doc_id for unsorted
    lists) and checks every question against the hash frozen in data/sample_ids.json.
    The original index is kept in the `orig_idx` column, so it appears in the logged samples.
    """
    items = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["items"]
    idx = [it["idx"] for it in items]
    subset = dataset.select(idx)
    for item, question in zip(items, subset["question"], strict=True):
        if _sha256(question) != item["question_sha256"]:
            raise ValueError(f"GSM8K question {item['idx']} differs from data/sample_ids.json")
    return subset.add_column("orig_idx", idx)


def doc_to_text_p0(doc) -> str:
    return P0_TEMPLATE.format(question=doc["question"])


def doc_to_text_p1(doc) -> str:
    return P1_TEMPLATE.format(question=doc["question"])
