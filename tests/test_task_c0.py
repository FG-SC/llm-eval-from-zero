"""The frozen sample and the C0 task (uses the GSM8K copy cached in .hf by `python -m src.sample`)."""
import json

from src.settings import ROOT, SAMPLE_FILE, prepare_environment
from tasks import utils

P0_EXPECTED = ("Solve the following math problem. Think step by step.\n"
               "On the last line, write the final answer as: #### <number>\n\n"
               "Problem: {q}")


def test_sample_file_is_consistent():
    data = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    idx = [it["idx"] for it in data["items"]]
    assert len(idx) == 100 == len(set(idx))
    assert idx == sorted(idx) == sorted(data["draw"]["order"])
    assert {f["idx"] for f in data["flagged"]} <= set(idx)
    assert SAMPLE_FILE.read_bytes().count(b"\r") == 0  # LF only, so the file hash is stable


def test_p0_prompt_is_exact():
    question = "Tom has {3} apples and buys 2 more. How many?"
    assert utils.doc_to_text_p0({"question": question}) == P0_EXPECTED.format(q=question)


def test_task_yields_the_100_sampled_questions_zero_shot():
    prepare_environment()
    from lm_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict(["gsm8k_seed100"], TaskManager(include_defaults=False, include_path=str(ROOT / "tasks")))[
        "gsm8k_seed100"]
    docs = list(task.test_docs())
    sample_idx = [it["idx"] for it in json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))["items"]]
    assert [d["orig_idx"] for d in docs] == sample_idx
    assert task.config.num_fewshot == 0
    gen = task.config.generation_kwargs
    assert gen["until"] == [] and gen["max_gen_toks"] == 2048 and gen["do_sample"] is False
    assert task.doc_to_text(docs[0]) == P0_EXPECTED.format(q=docs[0]["question"])
    assert [f.name for f in task._filters] == ["strict-match", "flexible-extract"]
