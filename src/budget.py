"""Quota guard: refuse a run whose estimate would push a model past a share of its daily quota.

Usage is read from this project's raw request logs (results/*/raw_requests.jsonl) for the last
24 hours. Other projects sharing the same Groq key are invisible here.
"""
import json
from datetime import datetime, timedelta, timezone

from src.settings import RESULTS_DIR

DEFAULT_TOKENS_PER_QUESTION = 2500  # conservative until a smoke test measures it


def usage_last_24h(model_id: str) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    used = {"requests": 0, "tokens": 0}
    for log in RESULTS_DIR.glob("*/raw_requests.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("model") != model_id or datetime.fromisoformat(rec["ts"]) < cutoff:
                continue
            used["requests"] += 1
            if rec.get("status") == 200:
                used["tokens"] += ((rec.get("response") or {}).get("usage") or {}).get("total_tokens") or 0
    return used


def measured_tokens_per_question(model_id: str) -> float | None:
    totals = []
    for log in RESULTS_DIR.glob("*/raw_requests.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("model") == model_id and rec.get("status") == 200:
                totals.append(((rec.get("response") or {}).get("usage") or {}).get("total_tokens") or 0)
    return sum(totals) / len(totals) if totals else None


def check(model_cfg: dict, n_questions: int, cap_fraction: float, cached: int = 0) -> tuple[bool, str]:
    """Plan only the questions without a cached complete answer."""
    limits = model_cfg["limits"]
    new = max(0, n_questions - cached)
    measured = measured_tokens_per_question(model_cfg["id"])
    per_q = measured * 1.5 if measured else DEFAULT_TOKENS_PER_QUESTION  # 50% margin on measurements
    planned_tokens = int(new * per_q)
    used = usage_last_24h(model_cfg["id"])
    token_cap = int(limits["tpd"] * cap_fraction)
    request_cap = int(limits["rpd"] * cap_fraction)
    ok = new == 0 or (used["tokens"] + planned_tokens <= token_cap and used["requests"] + new <= request_cap)
    basis = f"measured {measured:.0f}/question x1.5" if measured else f"default {DEFAULT_TOKENS_PER_QUESTION}/question"
    msg = (f"quota guard [{model_cfg['id']}]: used last 24h {used['tokens']:,} tokens / {used['requests']} requests; "
           f"this run: {n_questions} questions, {cached} already answered (cache), {new} to ask "
           f"~{planned_tokens:,} tokens ({basis}); "
           f"cap {cap_fraction:.0%} of daily = {token_cap:,} tokens / {request_cap} requests -> {'OK' if ok else 'REFUSED'}")
    return ok, msg
