"""Day 4 numbers from the judge's labels: counts, the flagged items kept apart, and the paired P0 -> P1 change."""
import json

from src import analyze

SAMPLE = {"flagged": [{"idx": 3, "variant": "critical thinking"}, {"idx": 9, "variant": "distraction insertion"}]}


def write_judgments(tmp_path, monkeypatch, table):
    """table: {run: [(orig_idx, rule_label, judge_label), ...]}"""
    folder = tmp_path / analyze.JUDGE_DIR
    folder.mkdir()
    lines = [json.dumps({"run": run, "orig_idx": i, "rule_label": rule, "judge_label": judge, "judge_raw": ""})
             for run, rows in table.items() for i, rule, judge in rows]
    (folder / "judgments.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(analyze, "RESULTS_DIR", tmp_path)


def test_no_judge_file_means_no_section(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "RESULTS_DIR", tmp_path)
    assert analyze.judge_section(SAMPLE) is None


def test_counts_pairs_and_the_flagged_item(tmp_path, monkeypatch):
    p0 = [(1, "answered_number", "ANSWERED_NUMBER"), (2, "no_final_answer", "ABSTAINED"),
          (3, "answered_number", "ANSWERED_NUMBER"), (4, "answered_number", "ANSWERED_NUMBER")]
    p1 = [(1, "abstained", "ABSTAINED"), (2, "abstained", "ABSTAINED"),
          (3, "answered_number", "ANSWERED_NUMBER"), (4, "abstained", "ABSTAINED")]
    write_judgments(tmp_path, monkeypatch, {"r3_c2_p0": p0, "r3_c2_p1": p1})
    out = analyze.judge_section(SAMPLE)
    r0, r1 = out["runs"]["r3_c2_p0"], out["runs"]["r3_c2_p1"]
    assert r0["n"] == 4 and r0["judge_labels"]["ANSWERED_NUMBER"]["correct"] == 3
    assert r0["judge_labels"]["ABSTAINED"]["correct"] == 1  # abstained in words: the rule label says no_final_answer
    assert r0["rule_vs_judge"]["no_final_answer -> ABSTAINED"] == 1
    assert r1["judge_labels"]["ABSTAINED"]["correct"] == 3
    # item 3 is a flagged item (it may be answerable): kept in 'all', left out of 'without_flagged'
    assert out["flagged_excluded"] == [3]
    assert r1["judge_labels_without_flagged"]["ABSTAINED"] == {**r1["judge_labels_without_flagged"]["ABSTAINED"],
                                                               "correct": 3, "n": 3}
    paired = out["abstained_p0_to_p1"]
    assert paired["all"]["n"] == 4 and paired["without_flagged"]["n"] == 3 and paired["excluded"] == [3]
    assert paired["all"]["before_only"] == 0 and paired["all"]["after_only"] == 2  # items 1 and 4 started to abstain
