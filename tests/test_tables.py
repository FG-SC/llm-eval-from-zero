"""What the notebook says in prose must stay true: grading-rule behaviour, the audit, the saved summary."""
import json
import re

import pytest

from src import analyze, tables
from src.settings import DATA_DIR


@pytest.fixture(scope="module")
def summary():
    return analyze.load()


def _platinum():
    return {it["idx"]: it for it in json.loads((DATA_DIR / "platinum_check.json").read_text(encoding="utf-8"))["items"]}


def _number(text):
    try:
        return float(str(text).replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def test_grading_demo_matches_the_notebook_text():
    rows = {r["Answer ends with"]: (r["Strict rule"], r["Flexible rule"]) for _, r in tables.grading_demo().iterrows()}
    assert rows["#### 64"] == ("reads 64 → right", "reads 64 → right")
    # "a dollar sign fools the strict one"
    assert rows["#### $64"] == ("reads [invalid] → wrong", "reads $64 → right")
    assert rows["####64"][0] == "reads [invalid] → wrong" and rows["####64"][1].endswith("right")
    assert rows["The answer is 64."][0] == "reads [invalid] → wrong" and rows["The answer is 64."][1].endswith("right")
    # "a note after the answer fools the flexible rule"
    assert rows["#### 64 (8 pairs of glasses)"] == ("reads 64 → right", "reads 8 → wrong")
    # "both reject 8.00 when the key says 8", and a right fraction fools both
    assert rows["#### 8.00"] == ("reads 8.00 → wrong", "reads 8.00 → wrong")
    assert rows["#### 10/2"] == ("reads 10 → wrong", "reads 2 → wrong")
    extra = tables.grading_demo([("#### 1,000", "1000")]).iloc[-1]
    assert extra["Case"] == "Yours" and extra["Strict rule"].endswith("right")


def test_answer_cards_keep_the_answer_line_visible(summary):
    cards = tables.answer_cards(summary)
    assert [c.split("\n", 1)[0].split(" · ")[0] for c in cards][-1] == "#### GSM8K #823"
    for card in cards:
        title, body = card.split("\n", 1)
        assert title.startswith("#### GSM8K #")
        outside_code = re.sub(r"```text\n.*?```", "", body, flags=re.DOTALL)
        assert not re.search(r"(?m)^\s*#", outside_code)  # '#### 64' must never become a heading
        assert "```text" in body  # the grader's view: raw text


def test_audit_covers_every_answer_graded_wrong(summary):
    a = summary["key_audit"]
    run = summary["runs"]["r1_c0"]
    wrong = sorted(r["orig_idx"] for r in run["per_question"] if not r["strict_ok"])
    assert sorted(i["idx"] for i in a["items"]) == wrong
    assert a["missing_from_audit"] == [] and sum(a["counts"].values()) == len(wrong)
    shown = {row["GSM8K #"]: row["Model wrote"] for _, row in tables.audit(summary).iterrows()}
    finals = {r["orig_idx"]: r["final_line"] for r in run["per_question"]}
    for idx, written in shown.items():  # what the model wrote, not what the strict rule read
        assert written in finals[idx] or "####" not in finals[idx]


def test_audit_readers_match_the_raw_verdicts(summary):
    raw = json.loads((DATA_DIR / "key_audit_readers.json").read_text(encoding="utf-8"))
    votes = {}
    for reader in raw["readers"]:
        for it in reader["items"]:
            votes.setdefault(it["idx"], {}).setdefault(it["label"].lower(), 0)
            votes[it["idx"]][it["label"].lower()] += 1
    for item in summary["key_audit"]["items"]:
        assert votes[item["idx"]] == item["votes"]
    assert summary["key_audit"]["n_readers"] == len(raw["readers"])


def test_platinum_steps_recomputed_independently(summary):
    plat, run = _platinum(), summary["runs"]["r1_c0"]
    rows = run["per_question"]
    kept = [r for r in rows if plat[r["orig_idx"]]["in_platinum"]]

    def right_with_platinum_key(r):
        p = plat[r["orig_idx"]]
        return r["strict_ok"] if not p["key_changed"] else analyze._as_key_text(r["strict"]) == p["platinum_key"]

    p = summary["key_audit"]["platinum"]
    assert p["steps"]["questions_only"]["correct"] == sum(r["strict_ok"] for r in kept)
    assert p["steps"]["questions_only"]["n"] == len(kept) == run["n"] - len(p["removed"])
    removed_right = sum(r["strict_ok"] for r in rows if not plat[r["orig_idx"]]["in_platinum"])
    assert p["steps"]["keys_only"]["correct"] == sum(right_with_platinum_key(r) for r in kept) + removed_right
    assert p["score"]["correct"] == sum(right_with_platinum_key(r) for r in kept)
    hidden = [r["orig_idx"] for r in rows if plat[r["orig_idx"]].get("key_changed") and r["strict_ok"]
              and not right_with_platinum_key(r)]
    assert p["hidden_mistakes"] == hidden


def test_human_review_is_counted_only_where_people_reviewed(summary):
    ag = summary["key_audit"]["platinum"]["agrees_with_audit"]
    plat = _platinum()
    checked = [i for i in summary["key_audit"]["items"] if i["verdict"] != "grader_format"]
    assert ag["reviewed_by_people"] == sum(plat[i["idx"]]["status"] != "consensus" for i in checked)
    assert ag["consensus"] == sorted(i["idx"] for i in checked if plat[i["idx"]]["status"] == "consensus")
    assert "reviewed by people" not in tables.PLATINUM_STATUS["consensus"]


def test_full_test_set_counts_match_the_platinum_card():
    full = json.loads((DATA_DIR / "platinum_check.json").read_text(encoding="utf-8"))["full_test_set"]
    # the dataset card's cleaning table: 1319 questions, 110 rejected, 10 re-labeled, 99 verified
    assert (full["n"], full["removed_from_platinum"], full["key_changed"], full["status_counts"]["verified"]) == \
        (1319, 110, 10, 99)


def test_summary_rebuilds_from_the_saved_results(summary):
    built = json.loads(json.dumps(analyze.build()))
    saved = dict(summary)
    built.pop("generated_at")
    saved.pop("generated_at")
    assert built == saved


def test_readme_results_table_is_up_to_date(summary):
    """The README's results table is generated from summary.json (python -m src.tables), never typed by hand."""
    block = tables.README_BLOCK.search(tables.README.read_text(encoding="utf-8"))
    assert block, "README.md has no <!-- results:start --> ... <!-- results:end --> block"
    assert block.group(0) == block.group(1) + tables.readme_block(summary) + block.group(2)
