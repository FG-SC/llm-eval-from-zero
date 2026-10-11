"""The judge's prompt, its JSON parsing and its quota guard (offline: the model is faked)."""
import json
import sys

import pytest

from src import judge


def test_prompt_has_rules_three_examples_and_the_answer_end():
    long_answer = "x" * (judge.TAIL_CHARS + 50) + "\n#### UNANSWERABLE"
    messages = judge.build_messages("How many pens?", long_answer)
    assert [m["role"] for m in messages] == ["system"] + ["user", "assistant"] * 3 + ["user"]
    assert all(json.loads(m["content"])["label"] in judge.LABELS for m in messages if m["role"] == "assistant")
    last = messages[-1]["content"]
    assert "How many pens?" in last and last.endswith("#### UNANSWERABLE") and "..." in last
    assert len(last) < judge.TAIL_CHARS + 100


@pytest.mark.parametrize("content, label", [
    ('{"label": "ABSTAINED"}', "ABSTAINED"),
    ('{"label": "ANSWERED_NUMBER"}', "ANSWERED_NUMBER"),
    ('{"label": "maybe"}', "INVALID"),
    ("ABSTAINED", "INVALID"),
    (None, "INVALID"),
])
def test_labels_are_validated(content, label):
    assert judge.parse_label(content) == label


def test_quota_guard_refusal_sends_nothing(monkeypatch, tmp_path):
    class Fake:
        def is_cached(self, messages, gen):
            return False

        def model_call(self, *a, **k):
            raise AssertionError("no request may be sent when the guard refuses")

    monkeypatch.setattr(judge, "make_judge", lambda raw_log: (Fake(), {}))
    monkeypatch.setattr(judge, "RESULTS_DIR", tmp_path)
    (tmp_path / "c2").mkdir()
    (tmp_path / "c2" / "samples_t.jsonl").write_text(json.dumps(
        {"doc": {"orig_idx": 1, "question": "q"}, "resps": [["#### 3"]], "answered_number": 1}) + "\n", encoding="utf-8")
    from src import budget
    monkeypatch.setattr(budget, "check", lambda *a, **k: (False, "stub: refused"))
    monkeypatch.setattr(sys, "argv", ["judge", "--run", "c2", "--name", "judge_test"])
    assert judge.main() == 2
    assert not (tmp_path / "judge_test").exists()
