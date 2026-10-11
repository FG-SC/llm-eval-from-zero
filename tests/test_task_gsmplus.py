"""GSM-Plus tasks: the same 100 seeds, the right variant type, and the critical-thinking classifier."""
import json

import pytest

from src.settings import ROOT, SAMPLE_FILE, prepare_environment
from tasks import gsmplus, utils

P1_EXPECTED = ("Solve the following math problem. Think step by step.\n"
               "On the last line, write the final answer as: #### <number>\n"
               "If the problem does not give enough information to compute a single answer, "
               "write on the last line: #### UNANSWERABLE\n\n"
               "Problem: {q}")


@pytest.mark.parametrize("text,label", [
    ("... so the total is 18.\n#### 18", "answered_number"),
    ("#### $1,200", "answered_number"),
    ("#### -5", "answered_number"),
    ("We cannot know the price.\n#### UNANSWERABLE", "abstained"),
    ("#### unanswerable", "abstained"),
    ("first guess #### 12 ... but data is missing\n#### UNANSWERABLE", "abstained"),
    ("The problem does not say how many eggs she sells.", "no_final_answer"),
    ("", "no_final_answer"),
    (None, "no_final_answer"),
])
def test_classify_final_line(text, label):
    assert gsmplus.classify_final_line(text) == label
    scores = gsmplus.process_results_critical({}, [text])
    assert scores[label] == 1 and sum(scores.values()) == 1


def test_p1_prompt_is_exact():
    assert utils.doc_to_text_p1({"question": "Q?"}) == P1_EXPECTED.format(q="Q?")


@pytest.mark.parametrize("task_name,perturbation,prompt", [
    ("gsmplus_distractor_seed100", "distraction insertion", "P0"),
    ("gsmplus_critical_seed100_p0", "critical thinking", "P0"),
    ("gsmplus_critical_seed100_p1", "critical thinking", "P1"),
])
def test_task_uses_the_same_100_seeds(task_name, perturbation, prompt):
    prepare_environment()
    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict([task_name], TaskManager(include_defaults=False, include_path=str(ROOT / "tasks")))[task_name]
    docs = list(task.test_docs())
    sample_idx = [it["idx"] for it in json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["items"]]
    assert [d["orig_idx"] for d in docs] == sample_idx
    assert {d["perturbation_type"] for d in docs} == {perturbation}
    assert task.config.num_fewshot == 0 and task.config.generation_kwargs["until"] == []
    template = utils.P0_TEMPLATE if prompt == "P0" else utils.P1_TEMPLATE
    assert task.doc_to_text(docs[0]) == template.format(question=docs[0]["question"])
    if perturbation == "critical thinking":
        assert {d["answer"] for d in docs} == {"None"}
