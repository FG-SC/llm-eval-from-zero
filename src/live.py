"""Day 2 demo: one question to the model, shown the way a student hands in an exam.

live=False shows a recorded request from the smoke run (free); live=True sends one real request
(about 350 tokens) with the same prompt and settings as the evaluation.
"""
import json

from src.settings import CONFIG_DIR, RESULTS_DIR, load_yaml, prepare_environment


def _p0(question: str) -> str:
    from tasks.utils import doc_to_text_p0
    return doc_to_text_p0({"question": question})


def _first_question() -> tuple[int, str, str]:
    path = RESULTS_DIR / "smoke_c0" / "samples_gsm8k_seed100.jsonl"
    s = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return s["doc"]["orig_idx"], s["doc"]["question"], str(s["target"]).split("####")[-1].strip()


def recorded() -> dict:
    idx, question, key = _first_question()
    raw = json.loads((RESULTS_DIR / "smoke_c0" / "raw_requests.jsonl").read_text(encoding="utf-8").splitlines()[0])
    choice = raw["response"]["choices"][0]
    return {"source": f"recorded on {raw['ts'][:10]}", "idx": idx, "question": question, "key": key,
            "answer": choice["message"]["content"], "reasoning": choice["message"].get("reasoning"),
            "finish_reason": choice["finish_reason"], "usage": raw["response"]["usage"], "seconds": raw.get("elapsed_s")}


def ask() -> dict:
    """One real request, through the same model class as the evaluation (paced, logged, metered)."""
    import time

    from lm_eval.models.api_models import JsonChatStr

    from src.groq_model import GroqChat

    prepare_environment()
    cfg = load_yaml(CONFIG_DIR / "models.yaml")["model_under_test"]
    model = GroqChat(model=cfg["id"], base_url=cfg["base_url"], rpm=cfg["limits"]["rpm"], tpm=cfg["limits"]["tpm"],
                     raw_log="results/live_demo/raw_requests.jsonl", seed=2026)  # no cache: really ask
    idx, question, key = _first_question()
    messages = [{"role": "user", "content": _p0(question)}]
    gen = {"until": [], "temperature": 0.0, "max_gen_toks": 2048, **(cfg.get("gen_kwargs") or {})}
    started = time.time()
    body = model.model_call([JsonChatStr(json.dumps(messages, ensure_ascii=False))], gen_kwargs=gen)
    choice = body["choices"][0]
    return {"source": "live request", "idx": idx, "question": question, "key": key,
            "answer": choice["message"]["content"], "reasoning": choice["message"].get("reasoning"),
            "finish_reason": choice["finish_reason"], "usage": body["usage"], "seconds": round(time.time() - started, 2)}


def show(live: bool = False) -> None:
    """The exact prompt, the model's reasoning text and visible answer, the raw line the grader reads,
    and how to read the numbers around them."""
    from IPython.display import Markdown, display

    from src.notebook import to_markdown

    r = ask() if live else recorded()
    u = r["usage"]
    reasoning_tokens = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    finished = "finished on its own" if r["finish_reason"] == "stop" else f"was cut off ({r['finish_reason']})"
    lines = [l for l in (r["answer"] or "").strip().splitlines() if l.strip()]
    plain = to_markdown(" ".join((r.get("reasoning") or "").split())).replace("*", "\\*").replace("_", "\\_")
    reasoning = (f"**Its reasoning text** (returned in a separate field; you pay for these tokens, but the grader "
                 f"does not read them):\n\n> {plain}\n\n" if r.get("reasoning") else "")
    display(Markdown(
        f"**{r['source'].capitalize()} · GSM8K #{r['idx']}**\n\n"
        f"**What we send** (one chat message; Groq wraps it in gpt-oss's chat format, a short header, before the "
        f"model reads it):\n\n```text\n{_p0(r['question'])}\n```\n\n"
        f"{reasoning}"
        f"**Its visible answer** (formulas rendered here):\n\n{to_markdown(r['answer'])}\n\n"
        f"**What the grader reads** (the raw last line):\n\n```text\n{lines[-1] if lines else ''}\n```\n\n"
        f"**Answer key:** {r['key']}\n\n"
        f"*How to read this:* the model {finished}. It read {u['prompt_tokens']} tokens (the chat format adds some to "
        f"our words) and wrote {u['completion_tokens']}, of which {reasoning_tokens} were reasoning text. The formulas "
        f"are written in LaTeX, a code for math: the grader reads the raw code, not the rendered version. "
        f"Time: {r['seconds']} s."))
