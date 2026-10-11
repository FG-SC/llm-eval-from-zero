"""Response cache that only keeps complete answers.

lm_eval's own cache (--use_cache) has two measured problems for this project: its key ignores the
model name, and it stores truncated or empty answers as final, so they are never asked again.
This cache keys on the prompt hash plus every request field (model, max_tokens, seed, ...) and stores
an answer only when every choice finished normally with non-empty content.

  python -m src.response_cache --rebuild    # refill the caches from results/*/raw_requests.jsonl
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def cache_key(prompt_hash: str, request: dict) -> str:
    blob = json.dumps({"prompt_hash": prompt_hash, "request": request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def usable(body: dict | None) -> bool:
    choices = (body or {}).get("choices") or []
    return bool(choices) and all(
        c.get("finish_reason") == "stop" and ((c.get("message") or {}).get("content") or "").strip()
        for c in choices)


class ResponseCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, prompt_hash TEXT, "
                         "model TEXT, response TEXT, stored_at TEXT)")
        self._db.commit()

    def get(self, key: str) -> dict | None:
        row = self._db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, prompt_hash: str, model: str, body: dict) -> bool:
        """Store only complete answers; returns whether it was stored."""
        if not usable(body):
            return False
        self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?)",
                         (key, prompt_hash, model, json.dumps(body, ensure_ascii=False),
                          datetime.now(timezone.utc).isoformat(timespec="seconds")))
        self._db.commit()
        return True

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._db.close()


def cache_path(root: Path, model_id: str) -> Path:
    return root / "cache" / model_id.replace("/", "__") / "responses.sqlite"


def _skipped_logs(root: Path) -> set[str]:
    """Logs that must not refill the cache: runs with cache: false (the Day 5 retest asked the same questions
    again on purpose; its answers would replace the Day 2 ones) and the notebook's one-off live demo."""
    runs_file = root / "configs" / "runs.yaml"
    if not runs_file.exists():
        return {"live_demo"}
    import yaml

    runs = yaml.safe_load(runs_file.read_text(encoding="utf-8")) or {}
    return {name for name, cfg in runs.items() if cfg.get("cache", True) is False} | {"live_demo"}


def rebuild(root: Path) -> dict:
    """Refill each model's cache from the raw request logs (successful, complete answers only)."""
    counts: dict[str, int] = {}
    caches: dict[str, ResponseCache] = {}
    skip = _skipped_logs(root)
    for log in sorted((root / "results").glob("*/raw_requests.jsonl")):
        if log.parent.name in skip:
            continue
        for line in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("status") != 200 or not rec.get("prompt_hash"):
                continue
            model = rec["model"]
            cache = caches.setdefault(model, ResponseCache(cache_path(root, model)))
            if cache.put(cache_key(rec["prompt_hash"], rec["request"]), rec["prompt_hash"], model, rec["response"]):
                counts[model] = counts.get(model, 0) + 1
    totals = {model: len(cache) for model, cache in caches.items()}
    for cache in caches.values():
        cache.close()
    return {"stored": counts, "entries": totals}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="refill caches from results/*/raw_requests.jsonl")
    args = ap.parse_args()
    if args.rebuild:
        print(json.dumps(rebuild(Path(__file__).resolve().parents[1]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
