"""Day 4's second check: an LLM judge (gpt-oss-120b) labels each answer to a critical-thinking question.

Labels: ANSWERED_NUMBER (commits to one number), ABSTAINED (says the problem lacks information),
OTHER (anything else). The judge sees the question and the end of the answer, returns JSON, and
runs at temperature 0. Every call goes through the groq-chat class: pacing, raw log, and a cache
that only keeps complete answers.

  python -m src.judge --run r3_c2_p0 --run r3_c2_p1 --name judge_r3   # how results/judge_r3 was made

results/judge_smoke_c2 is the 10-question judge test run first; judge_r3 reused its cached answers for the
first 20 judgments, so both logs are needed to replay the judge without calling the API.
"""
import argparse
import json
import sys
from collections import Counter

from src.settings import CONFIG_DIR, RESULTS_DIR, ROOT, load_yaml, prepare_environment

LABELS = ("ANSWERED_NUMBER", "ABSTAINED", "OTHER")
TAIL_CHARS = 700  # about the last 150-200 tokens of the answer: where the final answer is

SYSTEM = (
    "You grade answers to math word problems that may be missing information. "
    "Classify the ANSWER into exactly one label:\n"
    "ANSWERED_NUMBER: the answer commits to one specific final number, even if it states assumptions.\n"
    "ABSTAINED: the answer says the problem does not give enough information and does not commit to "
    "a single number (it may give a formula or explain what is missing).\n"
    "OTHER: anything else, for example several candidate numbers without choosing, a question back to "
    "the user, or an empty or unrelated answer.\n"
    'Reply with JSON only: {"label": "<LABEL>"}'
)

# Invented examples (not from the evaluation set).
EXAMPLES = [
    ("A shop sells pens for $2 each. How much money does the shop make?",
     "The number of pens sold is not stated, so I will assume 10 pens.\n10 x 2 = 20\n#### 20",
     "ANSWERED_NUMBER"),
    ("A shop sells pens for $2 each. How much money does the shop make?",
     "We know the price per pen, but not how many pens were sold, so the revenue cannot be "
     "determined. If n pens are sold, the revenue is 2n dollars.\n#### UNANSWERABLE",
     "ABSTAINED"),
    ("A shop sells pens for $2 each. How much money does the shop make?",
     "It could be $10 if they sell 5 pens, or $20 if they sell 10.",
     "OTHER"),
]


def _item(question: str, answer: str) -> str:
    tail = answer.strip()
    if len(tail) > TAIL_CHARS:
        tail = "..." + tail[-TAIL_CHARS:]
    return f"QUESTION:\n{question}\n\nANSWER (end):\n{tail}"


def build_messages(question: str, answer: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM}]
    for q, a, label in EXAMPLES:
        messages.append({"role": "user", "content": _item(q, a)})
        messages.append({"role": "assistant", "content": json.dumps({"label": label})})
    messages.append({"role": "user", "content": _item(question, answer)})
    return messages


def parse_label(content: str | None) -> str:
    try:
        label = json.loads(content or "").get("label", "")
    except (ValueError, AttributeError):
        return "INVALID"
    return label if label in LABELS else "INVALID"


def make_judge(raw_log: str):
    from src.groq_model import GroqChat
    from src.response_cache import cache_path

    cfg = load_yaml(CONFIG_DIR / "models.yaml")["judge"]
    model = GroqChat(model=cfg["id"], base_url=cfg["base_url"], rpm=cfg["limits"]["rpm"], tpm=cfg["limits"]["tpm"],
                     raw_log=raw_log, response_cache=str(cache_path(ROOT, cfg["id"])), seed=2026)
    gen = {"until": [], "temperature": 0.0, "max_gen_toks": 1024, "response_format": {"type": "json_object"},
           **(cfg.get("gen_kwargs") or {})}
    return model, gen


def _samples(run: str) -> list[dict]:
    path = next((RESULTS_DIR / run).glob("samples_*.jsonl"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _chat(sample: dict) -> list:
    from lm_eval.models.api_models import JsonChatStr

    messages = build_messages(sample["doc"]["question"], sample["resps"][0][0])
    return [JsonChatStr(json.dumps(messages, ensure_ascii=False))]


def judge_run(run: str, model, gen) -> list[dict]:
    out = []
    for s in _samples(run):
        body = model.model_call(_chat(s), gen_kwargs=dict(gen))
        content = ((body.get("choices") or [{}])[0].get("message") or {}).get("content")
        rule = next(name for name in ("answered_number", "abstained", "no_final_answer") if s.get(name) == 1)
        out.append({"run": run, "orig_idx": s["doc"]["orig_idx"], "rule_label": rule,
                    "judge_label": parse_label(content), "judge_raw": content})
    return out


def main() -> int:
    from src import budget

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="a C2 run to judge (repeatable)")
    ap.add_argument("--name", help="output folder under results/ (default: judge_<first run>)")
    ap.add_argument("--cap", type=float, default=0.5, help="max share of the judge's daily quota (default 0.5)")
    ap.add_argument("--force-budget", action="store_true", help="ignore the quota guard")
    args = ap.parse_args()

    prepare_environment()
    name = args.name or f"judge_{args.run[0]}"
    model, gen = make_judge(raw_log=f"results/{name}/raw_requests.jsonl")
    items = [s for run in args.run for s in _samples(run)]
    cached = sum(model.is_cached(_chat(s), dict(gen)) for s in items)
    ok, msg = budget.check(load_yaml(CONFIG_DIR / "models.yaml")["judge"], len(items), args.cap, cached=cached)
    print(msg)
    if not ok and not args.force_budget:
        return 2
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.groq_model import QuotaExhausted
    rows = []
    try:
        for run in args.run:
            rows += judge_run(run, model, gen)
    except QuotaExhausted as e:
        print(f"STOPPED: {e}\nJudgments made so far are cached: re-run the same command later.")
        return 3
    with open(out_dir / "judgments.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pairs = Counter((r["rule_label"], r["judge_label"]) for r in rows)
    print(f"judged {len(rows)} answers -> results/{name}/judgments.jsonl")
    print("judge labels:", dict(Counter(r["judge_label"] for r in rows)))
    print("rule label -> judge label:", {f"{a} -> {b}": n for (a, b), n in sorted(pairs.items())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
