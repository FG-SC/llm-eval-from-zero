"""The response cache keeps only complete answers and keys on the whole request (model included)."""
import json

from src.response_cache import ResponseCache, cache_key, rebuild, usable


def body(content="#### 18", finish="stop"):
    return {"choices": [{"index": 0, "message": {"content": content}, "finish_reason": finish}],
            "usage": {"total_tokens": 10}}


REQ = {"model": "m-a", "max_tokens": 2048, "temperature": 0.0, "stop": [], "seed": 2026, "n_messages": 1}


def test_only_complete_answers_are_usable():
    assert usable(body())
    assert not usable(body(finish="length"))
    assert not usable(body(content=""))
    assert not usable(body(content=None))
    assert not usable({"choices": []})


def test_put_get_and_refusals(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    k = cache_key("p1", REQ)
    assert cache.get(k) is None
    assert cache.put(k, "p1", "m-a", body())
    assert cache.get(k)["choices"][0]["message"]["content"] == "#### 18"
    truncated = cache_key("p2", REQ)
    assert not cache.put(truncated, "p2", "m-a", body(finish="length"))
    assert cache.get(truncated) is None and len(cache) == 1


def test_key_depends_on_model_and_settings():
    assert cache_key("p1", REQ) != cache_key("p1", {**REQ, "model": "m-b"})
    assert cache_key("p1", REQ) != cache_key("p1", {**REQ, "max_tokens": 512})
    assert cache_key("p1", REQ) != cache_key("p2", REQ)


def test_rebuild_from_raw_logs(tmp_path):
    run = tmp_path / "results" / "r"
    run.mkdir(parents=True)
    records = [
        {"status": 200, "model": "m-a", "prompt_hash": "p1", "request": REQ, "response": body()},
        {"status": 200, "model": "m-a", "prompt_hash": "p2", "request": REQ, "response": body(finish="length")},
        {"status": 429, "model": "m-a", "prompt_hash": "p3", "request": REQ, "response": {"error": {}}},
    ]
    (run / "raw_requests.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out = rebuild(tmp_path)
    assert out == {"stored": {"m-a": 1}, "entries": {"m-a": 1}}
