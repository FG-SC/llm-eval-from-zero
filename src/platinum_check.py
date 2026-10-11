"""Cross-check our 100 GSM8K questions against GSM8K-Platinum (Vendrow et al., 2025).

GSM8K-Platinum is a revised GSM8K test set. Its authors ran several frontier models on every
question and had people review each question that at least one model got wrong: they fixed wrong
answer keys and removed poorly written (mostly ambiguous) questions. Questions that every model
answered as the key says ("consensus") were kept without a human review. So it is a second,
partly human check of our own reading of the answers graded wrong, made with a different method.

Licences: the questions come from GSM8K (MIT); Platinum's annotations (cleaning status, revised
keys), which data/platinum_check.json repeats, are CC BY-SA 4.0.

  python -m src.platinum_check      # writes data/platinum_check.json
"""
import difflib
import json
import sys
from collections import Counter

from src.settings import DATA_DIR, GSM8K, SAMPLE_FILE, prepare_environment

PLATINUM = {"path": "madrylab/gsm8k-platinum", "name": "main", "split": "test",
            "revision": "e762492455a1cf7967de89f05b6bef72fc713b66"}
OUT_FILE = DATA_DIR / "platinum_check.json"


def final_answer(solution: str) -> str:
    return solution.split("####")[-1].strip().replace(",", "")


def match(gsm, plat, indices: list[int]) -> list[dict]:
    """For each GSM8K test index: kept or removed by Platinum, its status and both answer keys."""
    by_text = {q.strip(): i for i, q in enumerate(plat["question"])}
    plat_questions = [q.strip() for q in plat["question"]]
    items = []
    for idx in indices:
        q, key = gsm[idx]["question"].strip(), final_answer(gsm[idx]["answer"])
        j = by_text.get(q)
        reworded = False
        if j is None:  # the question text may have been edited: look for a near-identical one
            best = difflib.get_close_matches(q, plat_questions, n=1, cutoff=0.9)
            if best:
                j, reworded = plat_questions.index(best[0]), True
        if j is None:
            items.append({"idx": idx, "in_platinum": False, "status": "removed", "gsm8k_key": key})
            continue
        p_key = final_answer(plat[j]["answer"])
        items.append({"idx": idx, "in_platinum": True, "status": plat[j]["cleaning_status"],
                      "question_reworded": reworded, "gsm8k_key": key, "platinum_key": p_key,
                      "key_changed": p_key != key})
    return items


def counts(items: list[dict]) -> dict:
    return {"n": len(items), "removed_from_platinum": sum(not it["in_platinum"] for it in items),
            "key_changed": sum(it.get("key_changed", False) for it in items),
            "question_reworded": sum(it.get("question_reworded", False) for it in items),
            "status_counts": dict(Counter(it["status"] for it in items))}


def main() -> int:
    prepare_environment()
    from datasets import load_dataset

    gsm = load_dataset(GSM8K["path"], GSM8K["name"], split=GSM8K["split"], revision=GSM8K["revision"])
    plat = load_dataset(PLATINUM["path"], PLATINUM["name"], split=PLATINUM["split"], revision=PLATINUM["revision"])
    sample = [it["idx"] for it in json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["items"]]
    items = match(gsm, plat, sample)
    out = {"source": PLATINUM, "gsm8k": GSM8K, "n_sample": len(sample),
           "licence": "questions: GSM8K (MIT); Platinum annotations (status, revised keys): CC BY-SA 4.0",
           "summary": counts(items),
           "full_test_set": counts(match(gsm, plat, list(range(len(gsm))))),
           "items": items}
    with open(OUT_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"sample": out["summary"], "full_test_set": out["full_test_set"]}, indent=2))
    print(f"wrote {OUT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
