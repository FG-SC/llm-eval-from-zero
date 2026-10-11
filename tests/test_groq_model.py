"""Offline tests for the groq-chat model class (no network: requests.post is faked)."""
import json

import pytest
from lm_eval.models.api_models import JsonChatStr

from src import groq_model as gm

FAKE_KEY = "test-not-a-real-key-123"


class FakeResponse:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body


def answer(content="#### 18", finish="stop"):
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


def rate_limited(message, retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    return FakeResponse(429, {"error": {"message": message, "type": "tokens"}}, headers)


@pytest.fixture
def model(monkeypatch, tmp_path):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    return gm.GroqChat(model="fake-model", rpm=10**6, tpm=10**9, seed=2026, raw_log=str(tmp_path / "raw.jsonl"))


@pytest.fixture
def fake_post(monkeypatch):
    calls = []

    def install(*responses):
        queue = list(responses)

        def post(url, json=None, headers=None, **kwargs):
            calls.append({"url": url, "json": json, "headers": headers})
            return queue.pop(0)

        monkeypatch.setattr(gm.requests, "post", post)
        return calls

    return install


def chat():
    return [JsonChatStr(json.dumps([{"role": "user", "content": "What is 9 x 2?"}]))]


GEN = {"until": [], "do_sample": False, "temperature": 0.0, "max_gen_toks": 2048, "reasoning_effort": "low"}


def raw_lines(model):
    return [json.loads(l) for l in model.raw_log.read_text(encoding="utf-8").splitlines()]


def test_parse_duration():
    assert gm.parse_duration("1m26.4s") == pytest.approx(86.4)
    assert gm.parse_duration("11.32s") == pytest.approx(11.32)
    assert gm.parse_duration("952ms") == pytest.approx(0.952)
    assert gm.parse_duration("2h3m4s") == pytest.approx(7384)
    assert gm.parse_duration("7") == 7.0
    assert gm.parse_duration(None) is None


def test_truncated_or_empty_answers_become_none(model):
    assert model.parse_generations(answer("#### 18")) == ["#### 18"]
    assert model.parse_generations(answer("#### 1", finish="length")) == [None]
    assert model.parse_generations(answer("")) == [None]
    assert model.parse_generations(answer(None)) == [None]
    assert model.parse_generations({"choices": []}) == [None]


def test_payload_and_raw_log(model, fake_post):
    calls = fake_post(FakeResponse(200, answer(), {"x-ratelimit-remaining-tokens": "7000",
                                                   "x-ratelimit-reset-tokens": "1.5s", "x-request-id": "req_1"}))
    body = model.model_call(chat(), gen_kwargs=dict(GEN))
    assert body["choices"][0]["message"]["content"] == "#### 18"
    sent = calls[0]["json"]
    assert sent["max_tokens"] == 2048 and sent["stop"] == [] and sent["reasoning_effort"] == "low"
    assert sent["seed"] == 2026 and sent["temperature"] == 0.0
    assert calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    (rec,) = raw_lines(model)
    assert rec["status"] == 200 and rec["response"]["usage"]["total_tokens"] == 15
    assert rec["headers"]["x-request-id"] == "req_1"
    assert rec["prompt_hash"] and "messages" not in rec["request"]
    assert FAKE_KEY not in model.raw_log.read_text(encoding="utf-8")


def test_per_minute_429_is_retried(model, fake_post):
    calls = fake_post(
        rate_limited("Rate limit reached for model in organization org_abc123 on tokens per minute (TPM): "
                     "Please try again in 1.2s.", retry_after="1.2"),
        FakeResponse(200, answer()),
    )
    assert model.model_call(chat(), gen_kwargs=dict(GEN))["choices"]
    assert len(calls) == 2
    log = model.raw_log.read_text(encoding="utf-8")
    assert [r["status"] for r in raw_lines(model)] == [429, 200]
    assert "org_abc123" not in log and "org_REDACTED" in log
    assert FAKE_KEY not in log


def test_daily_limit_stops_the_run(model, fake_post):
    calls = fake_post(rate_limited("Rate limit reached on tokens per day (TPD): Limit 200000. "
                                   "Please try again in 1h2m3s."))
    with pytest.raises(gm.QuotaExhausted):
        model.model_call(chat(), gen_kwargs=dict(GEN))
    assert len(calls) == 1


def test_bad_request_is_not_retried(model, fake_post):
    calls = fake_post(FakeResponse(400, {"error": {"message": "`reasoning_effort` is not supported with this model"}}))
    with pytest.raises(gm.GroqAPIError, match="400"):
        model.model_call(chat(), gen_kwargs=dict(GEN))
    assert len(calls) == 1


def test_cache_serves_complete_answers_and_never_truncated(monkeypatch, tmp_path, fake_post):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gm.time, "sleep", lambda s: None)
    model = gm.GroqChat(model="fake-model", rpm=10**6, tpm=10**9, seed=2026, raw_log=str(tmp_path / "raw.jsonl"),
                        response_cache=str(tmp_path / "cache.sqlite"))
    calls = fake_post(FakeResponse(200, answer()), FakeResponse(200, answer("#### 1", finish="length")),
                      FakeResponse(200, answer("#### 1", finish="length")))
    model.model_call(chat(), gen_kwargs=dict(GEN))
    assert model.model_call(chat(), gen_kwargs=dict(GEN))["choices"][0]["message"]["content"] == "#### 18"
    assert len(calls) == 1 and model.cache_hits == 1  # second call served from the cache
    other = dict(GEN, max_gen_toks=512)  # different settings: not the same cached answer
    model.model_call(chat(), gen_kwargs=dict(other))
    model.model_call(chat(), gen_kwargs=dict(other))
    assert len(calls) == 3  # the truncated answer was not cached, so it was asked again
    # the quota guard counts cached answers with the same key, without sending anything
    assert model.is_cached(chat(), dict(GEN)) and not model.is_cached(chat(), dict(other))
    assert len(calls) == 3


def test_refuses_concurrency_and_missing_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    with pytest.raises(ValueError, match="num_concurrent"):
        gm.GroqChat(model="m", num_concurrent=2)
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        gm.GroqChat(model="m")
