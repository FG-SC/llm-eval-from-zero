"""Run one evaluation from configs/runs.yaml through lm-evaluation-harness (in-process).

  python -m src.run --list
  python -m src.run --run smoke_c0      # 10 of the 100 sampled questions
  python -m src.run --run r1_c0         # all 100

Writes results/<run>/: results.json, samples_<task>.jsonl, raw_requests.jsonl, run_summary.json.
Complete answers are cached per model in cache/<model>/responses.sqlite, so re-running costs nothing
except answers that were truncated or failed (those are asked again).
Exit codes: 0 ok, 2 refused by the quota guard, 3 daily quota hit (resume later).
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone

from src import budget
from src.settings import (CONFIG_DIR, RESULTS_DIR, ROOT, SAMPLE_FILE, load_yaml,
                          prepare_environment, sha256_file)


def ensure_utf8_mode() -> None:
    """lm_eval crashes printing its table under cp1252 (measured); re-launch in UTF-8 mode."""
    if not sys.flags.utf8_mode:
        sys.exit(subprocess.call([sys.executable, "-X", "utf8", "-m", "src.run", *sys.argv[1:]], cwd=ROOT))


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def write_json(path, obj, default=None) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, indent=2, default=default, ensure_ascii=False) + "\n")


def _signature(max_tokens, temperature, stop, reasoning_effort) -> tuple:
    return (max_tokens, float(temperature or 0), tuple(stop or ()), reasoning_effort)


def raw_index(model_id: str) -> dict:
    """(prompt_hash, generation signature) -> latest successful raw record, across all runs.

    Answers served from the cache were requested by an earlier run, so their usage and
    finish_reason live in that run's raw log.
    """
    records = []
    for log in RESULTS_DIR.glob("*/raw_requests.jsonl"):
        for line in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("model") == model_id and rec.get("status") == 200:
                records.append(rec)
    index = {}
    for rec in sorted(records, key=lambda r: r["ts"]):  # later answers win
        req = rec["request"]
        sig = _signature(req.get("max_tokens"), req.get("temperature"), req.get("stop"), req.get("reasoning_effort"))
        index[(rec["prompt_hash"], sig)] = rec
    return index


def cached_answers(model_id: str, task: str, gen: dict) -> int:
    """How many questions of this task already have a complete answer for these settings."""
    from src.response_cache import usable

    sig = _signature(gen.get("max_gen_toks"), gen.get("temperature"), gen.get("until"), gen.get("reasoning_effort"))
    index = raw_index(model_id)
    hashes = set()
    for path in RESULTS_DIR.glob(f"*/samples_{task}.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            ph = json.loads(line)["prompt_hash"]
            rec = index.get((ph, sig))
            if rec and usable(rec.get("response")):
                hashes.add(ph)
    return len(hashes)


def task_generation_kwargs(task: str, model_cfg: dict) -> dict:
    """The task YAML's generation settings plus the model's own fields (what the request will carry)."""
    from src.settings import load_task_yaml

    return {**load_task_yaml(task).get("generation_kwargs", {}), **(model_cfg.get("gen_kwargs") or {})}


def sample_signature(sample: dict) -> tuple:
    args = sample["arguments"]
    # in-process samples: [(prompt, gen_kwargs)]; samples written by lm_eval's CLI: {"gen_args_0": {"arg_1": ...}}
    gen = args["gen_args_0"]["arg_1"] if isinstance(args, dict) else args[0][1]
    return _signature(gen.get("max_gen_toks"), gen.get("temperature"), gen.get("until"), gen.get("reasoning_effort"))


def summarize(run_name, run_cfg, model_cfg, results, samples, raw_path, sample_sha, wall_s, session) -> dict:
    task = run_cfg["task"]
    index = raw_index(model_cfg["id"])
    statuses = Counter()
    requested_here = set()  # prompts sent to the API by THIS invocation (the log is append-only)
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("session") != session:
                continue
            statuses[str(rec.get("status"))] += 1
            requested_here.add(rec.get("prompt_hash"))

    by_doc = {}
    for s in samples.get(task, []):
        by_doc.setdefault(s["doc_id"], {})[s["filter"]] = s
    task_metrics = {k: v for k, v in results["results"][task].items() if k != "alias"}
    metric_names = sorted({k.split(",")[0] for k in task_metrics if "," in k and "_stderr" not in k})

    rows, prompt_toks, completion_toks, reasoning_toks = [], [], [], []
    finish = Counter()
    for doc_id in sorted(by_doc):
        per_filter = by_doc[doc_id]
        first = next(iter(per_filter.values()))
        raw = index.get((first.get("prompt_hash"), sample_signature(first)))
        usage = ((raw or {}).get("response") or {}).get("usage") or {}
        choice = (((raw or {}).get("response") or {}).get("choices") or [{}])[0]
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        if raw:
            prompt_toks.append(usage.get("prompt_tokens", 0))
            completion_toks.append(usage.get("completion_tokens", 0))
            reasoning_toks.append(reasoning or 0)
        finish[choice.get("finish_reason", "no_response")] += 1
        rows.append({
            "doc_id": doc_id,
            "orig_idx": first.get("doc", {}).get("orig_idx"),
            "gold": str(first.get("target", "")).split("####")[-1].strip(),
            # per scoring filter: what it extracted (None for the pass-through "none" filter) and its scores
            "filters": {name: {"extracted": s["filtered_resps"][0] if name != "none" else None,
                               "scores": {m: s[m] for m in metric_names if m in s}}
                        for name, s in per_filter.items()},
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning,
            "from_cache": first.get("prompt_hash") not in requested_here,
        })

    n = len(rows)
    unusable = sum(1 for r in rows if r["finish_reason"] != "stop")
    filter_names = [f for f in (rows[0]["filters"] if rows else {}) if f != "none"]

    def stats(values):
        return {"mean": round(statistics.mean(values), 1), "median": statistics.median(values),
                "max": max(values), "sum": sum(values)} if values else None

    return {
        "run": run_name, "purpose": run_cfg.get("purpose"), "task": task, "model": model_cfg["id"],
        "gen_kwargs": model_cfg.get("gen_kwargs"), "n_questions": n,
        "sample_ids_sha256": sample_sha,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "wall_seconds": round(wall_s, 1),
        "lm_eval_version": results.get("lm_eval_version"),
        "metrics": task_metrics,
        "extraction_rate": {f: sum(1 for r in rows if r["filters"][f]["extracted"] not in (None, "[invalid]")) / n
                            for f in filter_names} if n else {},
        "finish_reasons": dict(finish),
        "unusable_answers": {"count": unusable, "rate": unusable / n if n else None,
                             "note": "finish_reason != stop: truncated, empty or no response"},
        "tokens_per_question": {"prompt": stats(prompt_toks), "completion": stats(completion_toks),
                                "reasoning (inside completion)": stats(reasoning_toks)},
        "answers_from_cache": sum(1 for r in rows if r["from_cache"]),
        "http_status_counts_this_run": dict(statuses),
        "per_question": rows,
    }


def smoke_check(smoke: str) -> tuple[bool, str]:
    """A full run starts only after its smoke test passed: under 5% unusable answers (empty or cut off)
    and under 5% answers with nothing to grade (no number extracted, or no final line)."""
    path = RESULTS_DIR / smoke / "run_summary.json"
    if not path.exists():
        return False, f"smoke test {smoke} has not run yet"
    s = json.loads(path.read_text(encoding="utf-8"))
    unusable = s["unusable_answers"]["rate"]
    nothing = s["metrics"].get("no_final_answer,none")
    if nothing is None:
        nothing = 1 - min(s["extraction_rate"].values()) if s["extraction_rate"] else 1.0
    ok = unusable < 0.05 and nothing < 0.05
    return ok, (f"smoke test {smoke}: {unusable:.0%} unusable, {nothing:.0%} with nothing to grade "
                f"-> {'passed' if ok else 'FAILED (review it before any full run)'}")


def pending_full_runs(runs: dict) -> list[str]:
    """Full runs (limit: null) without results yet, in priority order."""
    full = [name for name, cfg in runs.items() if cfg["limit"] is None]
    return [name for name in sorted(full, key=lambda n: runs[n].get("priority", 99))
            if not (RESULTS_DIR / name / "results.json").exists()]


def next_run(runs: dict, models: dict, cap: float) -> str | None:
    """The first pending full run whose smoke test passed and whose estimate fits today's quota."""
    for name in pending_full_runs(runs):
        cfg = runs[name]
        passed, msg = smoke_check(cfg["smoke"])
        print(f"{name}: {msg}")
        if not passed:
            continue
        model_cfg = models[cfg["model"]]
        gen = task_generation_kwargs(cfg["task"], model_cfg)
        cached = min(100, cached_answers(model_cfg["id"], cfg["task"], gen)) if cfg.get("cache", True) else 0
        fits, msg = budget.check(model_cfg, 100, cap, cached=cached)
        print(f"{name}: {msg}")
        if fits:
            return name
    return None


def main() -> int:
    ensure_utf8_mode()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run name from configs/runs.yaml")
    ap.add_argument("--next", action="store_true",
                    help="run the next pending full run whose smoke test passed and that fits today's quota")
    ap.add_argument("--list", action="store_true", help="list the runs")
    ap.add_argument("--cap", type=float, default=0.5, help="max share of a model's daily quota (default 0.5)")
    ap.add_argument("--force-budget", action="store_true", help="ignore the quota guard")
    args = ap.parse_args()

    runs = load_yaml(CONFIG_DIR / "runs.yaml")
    models = load_yaml(CONFIG_DIR / "models.yaml")
    if args.next:
        os.chdir(ROOT)
        pending = pending_full_runs(runs)
        args.run = next_run(runs, models, args.cap) if pending else None
        if not args.run:
            print("no pending full run" if not pending else
                  f"nothing fits today (pending: {', '.join(pending)}); try again when the 24-hour window frees up")
            return 0 if not pending else 2
        print(f"next run: {args.run}")
    if args.list or not args.run:
        for name, cfg in runs.items():
            state = "done" if (RESULTS_DIR / name / "results.json").exists() else "pending"
            print(f"{name:13} {state:8} {models[cfg['model']]['id']:20} {cfg['task']:28} limit={cfg['limit']}  "
                  f"{cfg.get('purpose', '')}")
        return 0
    if args.run not in runs:
        print(f"unknown run {args.run!r}; see --list")
        return 1

    run_cfg = runs[args.run]
    model_cfg = models[run_cfg["model"]]
    os.chdir(ROOT)
    prepare_environment()

    n_questions = run_cfg["limit"] or 100
    use_cache = run_cfg.get("cache", True)  # cache: false = ask again (e.g. to measure run-to-run variation)
    gen = task_generation_kwargs(run_cfg["task"], model_cfg)
    cached = min(n_questions, cached_answers(model_cfg["id"], run_cfg["task"], gen)) if use_cache else 0
    ok, msg = budget.check(model_cfg, n_questions, args.cap, cached=cached)
    print(msg)
    if not ok and not args.force_budget:
        return 2

    from lm_eval import simple_evaluate
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable, make_table

    from src import groq_model  # registers "groq-chat"

    out_dir = RESULTS_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_rel = f"results/{args.run}/raw_requests.jsonl"
    sample_sha = sha256_file(SAMPLE_FILE)
    session = datetime.now(timezone.utc).isoformat(timespec="seconds")
    limits = model_cfg["limits"]
    model_args = {"model": model_cfg["id"], "base_url": model_cfg["base_url"], "rpm": limits["rpm"],
                  "tpm": limits["tpm"], "raw_log": raw_rel, "num_concurrent": 1, "seed": 2026, "session": session}
    if use_cache:
        model_args["response_cache"] = f"cache/{slug(model_cfg['id'])}/responses.sqlite"
    # lm_eval only keeps `metadata` if it is given to the TaskManager (simple_evaluate drops it otherwise)
    task_manager = TaskManager(include_defaults=False, include_path="tasks",
                               metadata={"sample_ids_sha256": f"sha256:{sample_sha}", "run": args.run})

    started = time.time()
    try:
        results = simple_evaluate(
            model="groq-chat", model_args=model_args, tasks=[run_cfg["task"]], limit=run_cfg["limit"],
            apply_chat_template=True, log_samples=True, gen_kwargs=dict(model_cfg.get("gen_kwargs") or {}),
            task_manager=task_manager,
        )
    except groq_model.QuotaExhausted as e:
        print(f"STOPPED: {e}\nAnswers received so far are cached: re-run the same command later to continue.")
        return 3
    wall = time.time() - started

    samples = results.pop("samples", {})
    write_json(out_dir / "results.json", results, default=handle_non_serializable)
    for task, rows in samples.items():
        with open(out_dir / f"samples_{task}.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, default=handle_non_serializable, ensure_ascii=False) + "\n")
    summary = summarize(args.run, run_cfg, model_cfg, results, samples, ROOT / raw_rel, sample_sha, wall, session)
    write_json(out_dir / "run_summary.json", summary)

    expected = ["results.json", f"samples_{run_cfg['task']}.jsonl", "run_summary.json"]
    missing = [name for name in expected if not (out_dir / name).exists()]
    print(make_table(results))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=2, ensure_ascii=False))
    if missing:
        print(f"ERROR: missing outputs: {missing}")
        return 1
    print(f"outputs: results/{args.run}/ ({', '.join(expected)}, raw_requests.jsonl)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
