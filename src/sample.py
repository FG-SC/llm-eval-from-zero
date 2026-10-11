"""Draw the 100 seed questions once and freeze them in data/sample_ids.json.

The same 100 GSM8K test questions are used in every condition (original, distractor,
critical thinking), which is what makes the comparisons paired.

  python -m src.sample            # create the file (refuses to overwrite)
  python -m src.sample --check    # re-draw and confirm the file still matches

The official sample hash (recorded in every result) is the sha256 of the file's bytes.
"""
import argparse
import json
import random
import sys

from src.settings import GSM8K, SAMPLE_FILE, prepare_environment, sha256_file, sha256_text

SEED = 2026
N = 100
DRAW_CODE = f"random.Random({SEED}).sample(range({GSM8K['n_rows']}), {N})"
# sha256(json.dumps(draw_order)) measured with Python 3.12.10 when the sample was drawn; guards against a
# Python version whose `random` gives a different draw.
EXPECTED_DRAW_SHA256 = "11e4cf1e597bad2b3a77e4d464700202275a6c77e7f9dbb81f40bb4c49d6e67d"

# Items with questionable gold labels found by reading the GSM-Plus variants before the runs.
# Recorded BEFORE any model run; results are reported with and without them.
FLAGGED = [
    {"idx": 1060, "variant": "critical thinking",
     "reason": "answerable: the removed sentence is never used in the seed solution, so gold 'None' is wrong (answer 31)"},
    {"idx": 1051, "variant": "critical thinking",
     "reason": "debatable: the removed fact (16 cups in a gallon) is common knowledge"},
    {"idx": 1122, "variant": "critical thinking",
     "reason": "debatable: says 'several workers' but asks about 'the fifth worker'"},
    {"idx": 980, "variant": "distraction insertion",
     "reason": "ambiguous: whether pearls count as beads changes the answer (90 vs 150)"},
    {"idx": 981, "variant": "distraction insertion",
     "reason": "ambiguous: 'that amount' has two possible referents"},
]


def draw() -> list[int]:
    order = random.Random(SEED).sample(range(GSM8K["n_rows"]), N)
    got = sha256_text(json.dumps(order))
    if got != EXPECTED_DRAW_SHA256:
        sys.exit(f"draw hash {got} != expected {EXPECTED_DRAW_SHA256}: this Python's random differs; "
                 "use Python 3.12 or the frozen file")
    return order


def build() -> dict:
    prepare_environment()
    from datasets import load_dataset

    ds = load_dataset(GSM8K["path"], GSM8K["name"], split=GSM8K["split"], revision=GSM8K["revision"])
    if len(ds) != GSM8K["n_rows"]:
        sys.exit(f"expected {GSM8K['n_rows']} test questions, got {len(ds)}")
    order = draw()
    questions = ds["question"]
    items = [{"idx": i, "question_sha256": sha256_text(questions[i])} for i in sorted(order)]
    sampled = {it["idx"] for it in items}
    missing = [f["idx"] for f in FLAGGED if f["idx"] not in sampled]
    if missing:
        sys.exit(f"flagged ids not in the sample: {missing}")
    return {
        "description": "The 100 GSM8K test questions used in every run (drawn once).",
        "dataset": GSM8K,
        "draw": {"code": DRAW_CODE, "python": "3.12", "order": order,
                 "order_sha256": EXPECTED_DRAW_SHA256},
        "items": items,
        "flagged": FLAGGED,
    }


def write(obj: dict) -> None:
    SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    with open(SAMPLE_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="re-draw and compare with the frozen file")
    args = ap.parse_args()

    if args.check:
        frozen = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
        fresh = build()
        same = frozen == fresh
        print(f"sample file matches a fresh draw: {same}")
        print(f"sample sha256 (file bytes): {sha256_file(SAMPLE_FILE)}")
        return 0 if same else 1

    if SAMPLE_FILE.exists():
        print(f"{SAMPLE_FILE.name} already exists; the sample is drawn once. Use --check to verify it.")
        return 1
    write(build())
    print(f"wrote {SAMPLE_FILE.relative_to(SAMPLE_FILE.parents[1])}: {N} questions, {len(FLAGGED)} flagged")
    print(f"sample sha256 (file bytes): {sha256_file(SAMPLE_FILE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
