"""Collect every finished run into results/summary.json: the single source of the numbers used
by the notebook, its charts (src/figures.py) and tables (src/tables.py). Reads the harness's sample logs, the raw request logs
and the data files data/key_audit.json (the AI readers' verdicts) and data/platinum_check.json (written by
src/platinum_check.py) directly.

  python -m src.analyze
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone

from src import stats
from src.run import _signature, sample_signature
from src.settings import CONFIG_DIR, DATA_DIR, RESULTS_DIR, SAMPLE_FILE, load_yaml, sha256_file

SUMMARY_FILE = RESULTS_DIR / "summary.json"

# Which run feeds which day of the notebook (smoke runs are checks, never results).
RUNS = {  # in the order of the days
    "r1_c0": {"condition": "C0", "label": "Original questions", "prompt": "P0", "kind": "numeric", "day": 2},
    "r1_c0_p1": {"condition": "C0", "label": "Original questions, may say UNANSWERABLE", "prompt": "P1",
                 "kind": "numeric", "day": 4},
    "r3_c2_p0": {"condition": "C2", "label": "Missing fact", "prompt": "P0", "kind": "abstention", "day": 4},
    "r3_c2_p1": {"condition": "C2", "label": "Missing fact, may say UNANSWERABLE", "prompt": "P1",
                 "kind": "abstention", "day": 4},
    "r1_c0_retest": {"condition": "C0", "label": "Original questions, asked again", "prompt": "P0",
                     "kind": "numeric", "day": 5, "retest_of": "r1_c0"},
    "r2_c1": {"condition": "C1", "label": "Extra irrelevant sentence", "prompt": "P0", "kind": "numeric", "day": 5},
}
RETEST_RUNS = {name for name, meta in RUNS.items() if meta.get("retest_of")}
CATEGORY_ORDER = ["right_both", "right_flexible_only", "right_strict_only", "right_but_missed_by_both", "wrong"]
CATEGORY_TEXT = {
    "right_both": "Right under both rules",
    "right_flexible_only": "Right, but only the flexible rule sees it",
    "right_strict_only": "Right, but only the strict rule sees it",
    "right_but_missed_by_both": "Right number, but both rules miss it",
    "wrong": "Doesn't match the key",
}
VERDICT_ORDER = ["model_error", "key_error", "grader_format", "ambiguous_question"]
VERDICT_TEXT = {
    "model_error": "The model's mistake",
    "key_error": "The answer key is wrong",
    "grader_format": "Right number, rejected by the grader",
    "ambiguous_question": "Ambiguous question",
}
LABEL_ORDER = ["answered_number", "abstained", "no_final_answer"]


# ---------------------------------------------------------------- helpers
def _number(text):
    if text is None or text == "[invalid]":
        return None
    try:
        return float(str(text).replace(",", "").replace("$", "").rstrip("."))
    except ValueError:
        return None


def _as_key_text(text) -> str:
    """The text exact_match compares: commas, '$' and one trailing period removed."""
    return str(text).replace(",", "").replace("$", "").removesuffix(".").strip()


def _tail(text: str, max_lines: int = 3) -> str:
    """The last lines of an answer, moved up when needed so that no formula is cut in half."""
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    start = max(0, len(lines) - max_lines)

    def unopened(part):
        return any(sum(l.count(close) for l in part) > sum(l.count(open_) for l in part)
                   for open_, close in (("\\[", "\\]"), ("\\(", "\\)")))

    while start > 0 and unopened(lines[start:]):
        start -= 1
    return "\n".join(lines[start:])


def _final_line(text: str) -> str:
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _written_answer(row: dict) -> str:
    """What the model wrote as its answer: the text after '####' on its last line (e.g. '10/3', which the
    strict rule reads as 10), or, without a '####' line, the last number in its answer."""
    m = re.search(r"####\s*(.+)$", row["final_line"])
    if m:
        return m.group(1).strip()
    return row["flexible"] if row["flexible"] != "[invalid]" else ""


def _load_samples(run: str, task: str) -> dict:
    by_doc = {}
    for line in (RESULTS_DIR / run / f"samples_{task}.jsonl").read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        by_doc.setdefault(s["doc_id"], {})[s["filter"]] = s
    return by_doc


def _raw_records(model_id: str, runs) -> list[dict]:
    records = []
    for run in runs:
        path = RESULTS_DIR / run / "raw_requests.jsonl"
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                if rec.get("model") == model_id and rec.get("status") == 200:
                    records.append(rec)
    return sorted(records, key=lambda r: r["ts"])


def _index(model_id: str, run: str) -> dict:
    """(prompt_hash, signature) -> raw record; the run's own requests win, then the other evaluation runs of
    configs/runs.yaml (never the retest, nor ad-hoc logs such as results/live_demo)."""
    evaluation_runs = load_yaml(CONFIG_DIR / "runs.yaml")
    others = [p.parent.name for p in RESULTS_DIR.glob("*/raw_requests.jsonl")
              if p.parent.name in evaluation_runs and p.parent.name != run and p.parent.name not in RETEST_RUNS]
    index = {}
    for rec in _raw_records(model_id, others) + _raw_records(model_id, [run]):
        req = rec["request"]
        index[(rec["prompt_hash"], _signature(req.get("max_tokens"), req.get("temperature"), req.get("stop"),
                                              req.get("reasoning_effort")))] = rec
    return index


def _usage(index: dict, sample: dict) -> dict:
    rec = index.get((sample["prompt_hash"], sample_signature(sample))) or {}
    resp = rec.get("response") or {}
    usage = resp.get("usage") or {}
    return {"prompt": usage.get("prompt_tokens"), "completion": usage.get("completion_tokens"),
            "reasoning": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "finish_reason": ((resp.get("choices") or [{}])[0]).get("finish_reason")}


def _tokens(per_doc: list[dict]) -> dict:
    def mean(key):
        vals = [d[key] for d in per_doc if d[key] is not None]
        return round(sum(vals) / len(vals), 1) if vals else None
    completions = [d["completion"] for d in per_doc if d["completion"] is not None]
    return {"prompt_mean": mean("prompt"), "completion_mean": mean("completion"),
            "reasoning_mean": mean("reasoning"), "completion_max": max(completions) if completions else None,
            "total": sum((d["prompt"] or 0) + (d["completion"] or 0) for d in per_doc),
            "finish_reasons": dict(Counter(d["finish_reason"] for d in per_doc))}


def _score(k: int, n: int) -> dict:
    lo, hi = stats.wilson(k, n)
    return {"correct": k, "n": n, "accuracy": k / n, "ci95": [lo, hi]}


def _card(results: dict, task: str, sample_file: dict) -> dict:
    """Everything needed to reproduce or compare the run (the 'rules of this exam')."""
    cfg = results["configs"][task]
    gen = cfg.get("generation_kwargs", {})
    model_args = results.get("config", {}).get("model_args", {})
    model_args = model_args if isinstance(model_args, dict) else {}
    return {"harness": f"lm-evaluation-harness {results.get('lm_eval_version')}",
            "task": task, "task_version": (cfg.get("metadata") or {}).get("version"),
            "dataset": f"{cfg.get('dataset_path')} {cfg.get('dataset_name') or ''} {cfg.get('test_split')}".replace("  ", " "),
            "dataset_revision": (cfg.get("dataset_kwargs") or {}).get("revision"),
            "draw": sample_file["draw"]["code"],
            "num_fewshot": cfg.get("num_fewshot"), "temperature": gen.get("temperature"),
            "max_gen_toks": gen.get("max_gen_toks"), "reasoning_effort": gen.get("reasoning_effort"),
            "stop": gen.get("until"), "api_seed": model_args.get("seed"), "chat_template": True}


# ---------------------------------------------------------------- run kinds
def numeric_run(run: str, meta: dict, model_id: str, sample_file: dict) -> dict:
    from tasks.gsmplus import classify_final_line

    results = json.loads((RESULTS_DIR / run / "results.json").read_text(encoding="utf-8"))
    task = next(iter(results["results"]))
    by_doc = _load_samples(run, task)
    index = _index(model_id, run)
    n = len(by_doc)
    rows, usage_rows = [], []
    for doc_id in sorted(by_doc):
        f = by_doc[doc_id]
        strict, flex = f["strict-match"], f["flexible-extract"]
        gold = str(strict["target"]).split("####")[-1].strip()
        s_ok, f_ok = bool(strict["exact_match"]), bool(flex["exact_match"])
        s_val, f_val = strict["filtered_resps"][0], flex["filtered_resps"][0]
        value = _number(s_val) if _number(s_val) is not None else _number(f_val)
        number_ok = value is not None and value == _number(gold)
        if s_ok and f_ok:
            cat = "right_both"
        elif f_ok:
            cat = "right_flexible_only"
        elif s_ok:
            cat = "right_strict_only"
        else:
            cat = "right_but_missed_by_both" if number_ok else "wrong"
        answer = strict["resps"][0][0]
        usage = _usage(index, strict)
        usage_rows.append(usage)
        rows.append({"orig_idx": strict["doc"]["orig_idx"], "question": strict["doc"]["question"], "gold": gold,
                     "category": cat, "strict": s_val, "strict_ok": s_ok, "flexible": f_val, "flexible_ok": f_ok,
                     "number_ok": bool(s_ok or f_ok or number_ok), "final_line": _final_line(answer),
                     "answer_tail": _tail(answer), "abstained": classify_final_line(answer) == "abstained",
                     "finish_reason": usage["finish_reason"]})

    harness = {name: {"exact_match": results["results"][task][f"exact_match,{name}"],
                      "stderr": results["results"][task][f"exact_match_stderr,{name}"]}
               for name in ("strict-match", "flexible-extract")}
    out = {**meta, "task": task, "n": n, "card": _card(results, task, sample_file),
           "scores": {"strict-match": _score(sum(r["strict_ok"] for r in rows), n),
                      "flexible-extract": _score(sum(r["flexible_ok"] for r in rows), n),
                      "number-match": _score(sum(r["number_ok"] for r in rows), n)},
           "harness": harness,
           "extracted": {"strict-match": sum(r["strict"] != "[invalid]" for r in rows),
                         "flexible-extract": sum(r["flexible"] != "[invalid]" for r in rows)},
           "categories": {c: sum(1 for r in rows if r["category"] == c) for c in CATEGORY_ORDER},
           "abstained": sum(r["abstained"] for r in rows),
           "tokens": _tokens(usage_rows),
           "unusable": sum(1 for u in usage_rows if u["finish_reason"] != "stop"),
           "per_question": rows}
    out["examples"] = _examples(rows)
    return out


def _examples(rows: list[dict]) -> list[dict]:
    """Real answers chosen by fixed rules, not by hand."""
    rules = [("clean", lambda r: r["category"] == "right_both"),
             ("number_not_text", lambda r: r["category"] == "right_but_missed_by_both"),
             ("no_box", lambda r: r["strict"] == "[invalid]"),
             ("rules_disagree", lambda r: r["strict"] != "[invalid]" and r["strict"] != r["flexible"])]
    picked, used = [], set()
    for kind, rule in rules:
        hit = next((r for r in rows if rule(r) and r["orig_idx"] not in used), None)
        if hit:
            used.add(hit["orig_idx"])
            picked.append({"kind": kind, **{k: hit[k] for k in ("orig_idx", "question", "gold", "strict", "strict_ok",
                                                                  "flexible", "flexible_ok", "final_line", "answer_tail")}})
    return picked


def _key_error_example(run: dict, audit: dict | None) -> list[dict]:
    """A fifth answer card, also picked by a rule: the first answer whose official key the audit found wrong."""
    item = next((i for i in sorted((audit or {}).get("items", []), key=lambda i: i["idx"])
                 if i["verdict"] == "key_error"), None)
    if not item:
        return []
    r = next(r for r in run["per_question"] if r["orig_idx"] == item["idx"])
    return [{"kind": "key_error", **{k: r[k] for k in ("orig_idx", "question", "gold", "strict", "strict_ok", "flexible",
                                                       "flexible_ok", "final_line", "answer_tail")},
             "platinum_key": item["platinum_key"], "reason": item["reason"]}]


def key_audit(run: dict) -> dict | None:
    """Day 3: the post-hoc reading of the answers graded wrong, plus the GSM8K-Platinum cross-check."""
    path, plat_path = DATA_DIR / "key_audit.json", DATA_DIR / "platinum_check.json"
    if not path.exists():
        return None
    audit = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["orig_idx"]: r for r in run["per_question"]}
    plat_file = json.loads(plat_path.read_text(encoding="utf-8")) if plat_path.exists() else {}
    plat = {it["idx"]: it for it in plat_file.get("items", [])}
    graded_wrong = [r["orig_idx"] for r in run["per_question"] if not r["strict_ok"]]
    items = []
    for it in audit["items"]:
        r = rows[it["idx"]]
        p = plat.get(it["idx"], {})
        items.append({**it, "question": r["question"], "gold": r["gold"], "model_final_line": r["final_line"],
                      "model_answer": _written_answer(r), "grader_read": r["strict"],
                      "platinum": p.get("status"), "platinum_key": p.get("platinum_key")})
    out = {"readers": audit["readers"], "date": audit["date"], "graded_wrong": len(graded_wrong),
           "n_readers": max(sum(i["votes"].values()) for i in items) if items else 0,
           "audited": len(items), "missing_from_audit": sorted(set(graded_wrong) - {i["idx"] for i in items}),
           "counts": {v: sum(1 for i in items if i["verdict"] == v) for v in VERDICT_ORDER},
           "unanimous": sum(1 for i in items if len(i["votes"]) == 1), "items": items}
    if plat:
        notes = {n["idx"]: n["note"] for n in audit.get("key_change_notes", {}).get("items", [])}
        rows_in = [r for r in run["per_question"] if plat[r["orig_idx"]]["in_platinum"]]

        def right_with_new_key(r):
            p = plat[r["orig_idx"]]
            return r["strict_ok"] if not p.get("key_changed") else _as_key_text(r["strict"]) == _as_key_text(p["platinum_key"])

        changed = [{"idx": r["orig_idx"], "gsm8k_key": plat[r["orig_idx"]]["gsm8k_key"],
                    "platinum_key": plat[r["orig_idx"]]["platinum_key"], "model": r["strict"],
                    "right_before": r["strict_ok"], "right_after": bool(right_with_new_key(r)),
                    "note": notes.get(r["orig_idx"])}
                   for r in run["per_question"] if plat[r["orig_idx"]].get("key_changed")]
        removed = [r for r in run["per_question"] if not plat[r["orig_idx"]]["in_platinum"]]
        n = len(run["per_question"])
        out["platinum"] = {
            "kept": len(rows_in), "removed": sorted(r["orig_idx"] for r in removed),
            "removed_right": sorted(r["orig_idx"] for r in removed if r["strict_ok"]),
            "removed_wrong": sorted(r["orig_idx"] for r in removed if not r["strict_ok"]),
            "key_changed": changed,
            "hidden_mistakes": [c["idx"] for c in changed if c["right_before"] and not c["right_after"]],
            "agrees_with_audit": _platinum_agreement(items),
            # the same answers, graded step by step: corrected keys only, removed questions only, both
            "steps": {"keys_only": _score(sum(right_with_new_key(r) for r in run["per_question"] if plat[r["orig_idx"]]["in_platinum"])
                                          + sum(r["strict_ok"] for r in removed), n),
                      "questions_only": _score(sum(r["strict_ok"] for r in rows_in), len(rows_in))},
            "score": _score(sum(right_with_new_key(r) for r in rows_in), len(rows_in)),
            "full_test_set": plat_file.get("full_test_set"),
        }
    return out


def _platinum_agreement(items: list[dict]) -> dict:
    """Does GSM8K-Platinum agree with each audit verdict? People reviewed only the questions its models got
    wrong ('verified', 'revised', 'removed'); 'consensus' questions were kept without a human review."""
    expected = {"model_error": lambda p: p["platinum"] in ("consensus", "verified") and p["platinum_key"] == p["gold"],
                "key_error": lambda p: p["platinum"] == "revised",
                "ambiguous_question": lambda p: p["platinum"] == "removed"}
    checked = [i for i in items if i["verdict"] in expected]
    agree = [i["idx"] for i in checked if expected[i["verdict"]](i)]
    return {"checked": len(checked), "agree": len(agree), "disagree": sorted({i["idx"] for i in checked} - set(agree)),
            "reviewed_by_people": sum(1 for i in checked if i["platinum"] in ("verified", "revised", "removed")),
            "consensus": sorted(i["idx"] for i in checked if i["platinum"] == "consensus"),
            "not_compared": [{"idx": i["idx"], "verdict": i["verdict"], "platinum": i["platinum"]}
                             for i in items if i["verdict"] not in expected]}


def abstention_run(run: str, meta: dict, model_id: str, sample_file: dict) -> dict:
    results = json.loads((RESULTS_DIR / run / "results.json").read_text(encoding="utf-8"))
    task = next(iter(results["results"]))
    by_doc = _load_samples(run, task)
    index = _index(model_id, run)
    flagged = {f["idx"] for f in sample_file["flagged"] if f["variant"] == "critical thinking"}
    rows, usage_rows = [], []
    for doc_id in sorted(by_doc):
        s = next(iter(by_doc[doc_id].values()))
        usage = _usage(index, s)
        usage_rows.append(usage)
        rows.append({"orig_idx": s["doc"]["orig_idx"], "question": s["doc"]["question"],
                     "rule_label": next(name for name in LABEL_ORDER if s.get(name) == 1),
                     "final_line": _final_line(s["resps"][0][0]), "answer_tail": _tail(s["resps"][0][0]),
                     "finish_reason": usage["finish_reason"]})

    def labels(subset):
        n = len(subset)
        return {name: {**_score(sum(1 for r in subset if r["rule_label"] == name), n)} for name in LABEL_ORDER}

    return {**meta, "task": task, "n": len(rows), "card": _card(results, task, sample_file),
            "rule_labels": labels(rows),
            "rule_labels_without_flagged": labels([r for r in rows if r["orig_idx"] not in flagged]),
            "flagged_excluded": sorted(flagged & {r["orig_idx"] for r in rows}),
            "tokens": _tokens(usage_rows), "unusable": sum(1 for u in usage_rows if u["finish_reason"] != "stop"),
            "per_question": rows}


# ---------------------------------------------------------------- comparisons
def _paired(a: dict, b: dict, ids: list[int]) -> dict:
    x, y = [a[i] for i in ids], [b[i] for i in ids]
    only_a = sum(1 for i, j in zip(x, y) if i and not j)
    only_b = sum(1 for i, j in zip(x, y) if j and not i)
    diff, lo, hi = stats.bootstrap_paired_diff(x, y)
    return {"n": len(ids), "rate_before": sum(x) / len(ids), "rate_after": sum(y) / len(ids),
            "before_only": only_a, "after_only": only_b, "mcnemar_p": stats.mcnemar_exact(only_a, only_b),
            "diff": diff, "diff_ci95_bootstrap": [lo, hi]}


def compare_numeric(before: dict, after: dict, exclude: set) -> dict:
    """Same questions, two runs: per question 'right' (strict-match), compared pair by pair."""
    a = {r["orig_idx"]: int(r["strict_ok"]) for r in before["per_question"]}
    b = {r["orig_idx"]: int(r["strict_ok"]) for r in after["per_question"]}
    ids = sorted(set(a) & set(b))
    return {"all": _paired(a, b, ids), "without_flagged": _paired(a, b, [i for i in ids if i not in exclude]),
            "excluded": sorted(exclude & set(ids))}


def compare_abstention(p0: dict, p1: dict, exclude: set) -> dict:
    """Day 4 headline: on the same questions, does permission to abstain change the answers?"""
    a = {r["orig_idx"]: int(r["rule_label"] == "abstained") for r in p0["per_question"]}
    b = {r["orig_idx"]: int(r["rule_label"] == "abstained") for r in p1["per_question"]}
    ids = sorted(set(a) & set(b))
    return {"all": _paired(a, b, ids), "without_flagged": _paired(a, b, [i for i in ids if i not in exclude]),
            "excluded": sorted(exclude & set(ids))}


JUDGE_LABELS = ["ANSWERED_NUMBER", "ABSTAINED", "OTHER", "INVALID"]
JUDGE_DIR = "judge_r3"  # python -m src.judge --run r3_c2_p0 --run r3_c2_p1 --name judge_r3


def judge_section(sample_file: dict) -> dict | None:
    """Day 4: what the LLM judge says about the answers to the missing-fact exam. Without the permission to
    write UNANSWERABLE, a model can abstain in plain words, which a rule on the last line cannot see."""
    path = RESULTS_DIR / JUDGE_DIR / "judgments.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    flagged = {f["idx"] for f in sample_file["flagged"] if f["variant"] == "critical thinking"}

    def counts(subset):
        return {lab: _score(sum(1 for r in subset if r["judge_label"] == lab), len(subset)) for lab in JUDGE_LABELS}

    per_run = {}
    for run in sorted({r["run"] for r in rows}):
        mine = [r for r in rows if r["run"] == run]
        per_run[run] = {"n": len(mine), "judge_labels": counts(mine),
                        "judge_labels_without_flagged": counts([r for r in mine if r["orig_idx"] not in flagged]),
                        "rule_vs_judge": {f"{a} -> {b}": c for (a, b), c in
                                          sorted(Counter((r["rule_label"], r["judge_label"]) for r in mine).items())}}
    agree = {("answered_number", "ANSWERED_NUMBER"), ("abstained", "ABSTAINED")}
    out = {"judge_dir": JUDGE_DIR, "flagged_excluded": sorted(flagged), "runs": per_run,
           "disagreements": [{"run": r["run"], "idx": r["orig_idx"], "rule": r["rule_label"], "judge": r["judge_label"]}
                             for r in rows if (r["rule_label"], r["judge_label"]) not in agree]}
    if {"r3_c2_p0", "r3_c2_p1"} <= set(per_run):
        a = {r["orig_idx"]: int(r["judge_label"] == "ABSTAINED") for r in rows if r["run"] == "r3_c2_p0"}
        b = {r["orig_idx"]: int(r["judge_label"] == "ABSTAINED") for r in rows if r["run"] == "r3_c2_p1"}
        ids = sorted(set(a) & set(b))
        out["abstained_p0_to_p1"] = {"all": _paired(a, b, ids),
                                     "without_flagged": _paired(a, b, [i for i in ids if i not in flagged]),
                                     "excluded": sorted(flagged & set(ids))}
    return out


def retest(original: dict, again: dict) -> dict:
    """The same questions asked twice: how many final answers changed, and how many grades flipped?"""
    a = {r["orig_idx"]: r for r in original["per_question"]}
    b = {r["orig_idx"]: r for r in again["per_question"]}
    ids = sorted(set(a) & set(b))
    changed = [i for i in ids if _as_key_text(a[i]["strict"]) != _as_key_text(b[i]["strict"])]
    flips = [i for i in ids if a[i]["strict_ok"] != b[i]["strict_ok"]]
    return {"n": len(ids), "answer_changed": _score(len(changed), len(ids)), "grade_flipped": _score(len(flips), len(ids)),
            "changed_idx": changed, "flipped_idx": flips,
            "right_to_wrong": sum(1 for i in flips if a[i]["strict_ok"]), "wrong_to_right": sum(1 for i in flips if not a[i]["strict_ok"])}


# ---------------------------------------------------------------- build
def build() -> dict:
    models = load_yaml(CONFIG_DIR / "models.yaml")
    model_id = models["model_under_test"]["id"]
    sample_file = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "model": model_id, "judge": models["judge"]["id"],
           "sample": {"sha256": sha256_file(SAMPLE_FILE), "n": len(sample_file["items"]), "dataset": sample_file["dataset"],
                      "draw": sample_file["draw"]["code"], "flagged": sample_file["flagged"]},
           "runs": {}, "comparisons": {}}
    for run, meta in RUNS.items():
        if not (RESULTS_DIR / run / "results.json").exists():
            continue
        summary = json.loads((RESULTS_DIR / run / "run_summary.json").read_text(encoding="utf-8"))
        meta = {**meta, "date": summary["finished_at"][:10]}
        builder = numeric_run if meta["kind"] == "numeric" else abstention_run
        out["runs"][run] = builder(run, meta, model_id, sample_file)

    runs = out["runs"]
    if "r1_c0" in runs:
        out["key_audit"] = key_audit(runs["r1_c0"])
        runs["r1_c0"]["examples"] += _key_error_example(runs["r1_c0"], out["key_audit"])
    key_errors = {i["idx"] for i in (out.get("key_audit") or {}).get("items", []) if i["verdict"] == "key_error"}
    key_errors |= {c["idx"] for c in ((out.get("key_audit") or {}).get("platinum") or {}).get("key_changed", [])}
    di_flags = {f["idx"] for f in sample_file["flagged"] if f["variant"] == "distraction insertion"}
    ct_flags = {f["idx"] for f in sample_file["flagged"] if f["variant"] == "critical thinking"}
    if "r1_c0" in runs and "r2_c1" in runs:
        out["comparisons"]["c0_to_c1"] = compare_numeric(runs["r1_c0"], runs["r2_c1"], di_flags | key_errors)
    if "r1_c0" in runs and "r1_c0_retest" in runs:
        out["comparisons"]["c0_retest"] = retest(runs["r1_c0"], runs["r1_c0_retest"])
        out["comparisons"]["c0_to_retest"] = compare_numeric(runs["r1_c0"], runs["r1_c0_retest"], key_errors)
    if "r1_c0" in runs and "r1_c0_p1" in runs:
        out["comparisons"]["c0_to_p1"] = compare_numeric(runs["r1_c0"], runs["r1_c0_p1"], key_errors)
    if "r3_c2_p0" in runs and "r3_c2_p1" in runs:
        out["comparisons"]["c2_p0_to_p1"] = compare_abstention(runs["r3_c2_p0"], runs["r3_c2_p1"], ct_flags)
    if judge := judge_section(sample_file):
        out["judge_results"] = judge
    return out


def write(summary: dict) -> None:
    with open(SUMMARY_FILE, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")


def load() -> dict:
    """The committed summary; rebuild it if it does not exist yet."""
    if not SUMMARY_FILE.exists():
        write(build())
    return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))


def main() -> int:
    summary = build()
    write(summary)
    for run, r in summary["runs"].items():
        if r["kind"] == "numeric":
            s, m = r["scores"]["strict-match"], r["scores"]["number-match"]
            print(f"{run:14} {r['condition']} {r['prompt']} n={r['n']} strict {s['correct']}/{r['n']} "
                  f"[{s['ci95'][0]:.1%}, {s['ci95'][1]:.1%}]  number-match {m['correct']}  categories {r['categories']}")
        else:
            labs = {k: v["correct"] for k, v in r["rule_labels"].items()}
            print(f"{run:14} {r['condition']} {r['prompt']} n={r['n']} rule labels {labs}")
    audit = summary.get("key_audit")
    if audit:
        print(f"key audit: {audit['counts']}  platinum: {audit.get('platinum', {}).get('score')}"
              f"  agreement: {audit.get('platinum', {}).get('agrees_with_audit')}")
    print(f"wrote {SUMMARY_FILE.relative_to(SUMMARY_FILE.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
