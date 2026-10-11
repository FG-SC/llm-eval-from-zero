"""lm_eval model class `groq-chat`: Groq's OpenAI-compatible chat endpoint.

It extends lm_eval's `local-chat-completions` with what a first test run showed is missing:
- reads GROQ_API_KEY (never written anywhere);
- paces requests by the per-minute limits and honours Retry-After on HTTP 429;
- stops cleanly with QuotaExhausted when a daily limit is hit (the response cache keeps progress);
- appends one raw JSON line per HTTP attempt: usage, finish_reason, rate-limit headers, full body,
  tagged with the session (one process run) that made it;
- has its own response cache (src/response_cache.py) keyed on the prompt and every request field,
  which stores only complete answers. lm_eval's --use_cache is NOT used: it ignores the model name
  and stores truncated answers as final (both measured);
- returns None for truncated (finish_reason="length") or empty answers, so they are scored as
  invalid, never cached, and asked again on the next run.
Only num_concurrent=1: lm_eval's async path logs the Authorization header on errors.
"""
import copy
import hashlib
import json
import os
import re
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import requests
from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import postprocess_generated_text

from src.response_cache import ResponseCache, cache_key

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
KEEP_HEADERS = ("x-ratelimit", "retry-after", "x-request-id", "x-groq")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_TRY_AGAIN = re.compile(r"try again in ([0-9hms.]+)", re.IGNORECASE)
_ORG_ID = re.compile(r"org_[A-Za-z0-9]+")


class QuotaExhausted(RuntimeError):
    """A daily request/token limit was hit; resume later (cached answers are kept)."""


class GroqAPIError(RuntimeError):
    """A non-retryable API error (e.g. HTTP 400) or retries exhausted."""


def parse_duration(text: str | None) -> float | None:
    """Groq durations: '7', '11.32s', '952ms', '1m26.4s', '2h3m4s' -> seconds."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if not parts:
        return None
    scale = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    return sum(float(n) * scale[u] for n, u in parts)


def _error_message(body) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message", ""))
        if err:
            return str(err)
    return ""


@register_model("groq-chat")
class GroqChat(LocalChatCompletion):
    def __init__(
        self,
        base_url: str = GROQ_CHAT_URL,
        rpm: int = 30,
        tpm: int = 8000,
        raw_log: str | None = None,
        response_cache: str | None = None,
        session: str | None = None,
        daily_wait_limit_s: float = 120.0,
        max_attempts: int = 8,
        **kwargs,
    ):
        kwargs.setdefault("max_retries", 1)  # retries happen here, not in lm_eval's wrapper
        kwargs.setdefault("tokenized_requests", False)
        super().__init__(base_url=base_url, **kwargs)
        if self._concurrent > 1:
            raise ValueError("groq-chat needs num_concurrent=1 (lm_eval's async path logs the API key on errors)")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set (put it in .env)")
        self.rpm = int(rpm)
        self.tpm = int(tpm)
        self.raw_log = Path(raw_log) if raw_log else None
        self.cache = ResponseCache(response_cache) if response_cache else None
        self.session = session or datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.cache_hits = 0
        self.daily_wait_limit_s = float(daily_wait_limit_s)
        self.max_attempts = int(max_attempts)
        self._window: deque[tuple[float, int]] = deque()  # (time, total_tokens) of the last 60 s
        self._last_start = 0.0
        self._server_tokens: tuple[float, int, float] | None = None  # (received_at, remaining, reset_s)
        self._recent_totals: deque[int] = deque(maxlen=20)
        self._seq = 0

    @property
    def api_key(self) -> str:
        return os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY", "")

    # ---------- pacing ----------
    def _expected_tokens(self, payload: dict) -> int:
        prompt_estimate = len(json.dumps(payload.get("messages", []), ensure_ascii=False)) // 3
        typical = int(sum(self._recent_totals) / len(self._recent_totals)) if self._recent_totals else 1500
        return max(prompt_estimate, 1) + typical

    def _pace(self, payload: dict) -> None:
        now = time.time()
        waits = [self._last_start + 60.0 / self.rpm * 1.05 - now]
        while self._window and now - self._window[0][0] > 60.0:
            self._window.popleft()
        need = self._expected_tokens(payload)
        budget = int(self.tpm * 0.9)
        used = sum(t for _, t in self._window)
        if self._window and used + need > budget:
            freed = 0
            for t0, tokens in self._window:
                freed += tokens
                if used - freed + need <= budget:
                    waits.append(t0 + 60.0 - now)
                    break
            else:
                waits.append(self._window[-1][0] + 60.0 - now)
        if self._server_tokens:
            received_at, remaining, reset_s = self._server_tokens
            if remaining < need:
                waits.append(received_at + reset_s - now)
        wait = max(waits + [0.0])
        if wait > 0:
            time.sleep(wait)
        self._last_start = time.time()

    def _note_headers(self, headers: dict) -> None:
        remaining = headers.get("x-ratelimit-remaining-tokens")
        reset_s = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if remaining is not None and reset_s is not None:
            self._server_tokens = (time.time(), int(remaining), reset_s)

    # ---------- raw log ----------
    def _log(self, record: dict) -> None:
        if not self.raw_log:
            return
        self.raw_log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False)
        line = _ORG_ID.sub("org_REDACTED", line)
        with open(self.raw_log, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")

    # ---------- the request ----------
    def _prepare(self, messages, gen_kwargs, generate=True, **kwargs):
        """The request body (built exactly as lm_eval's local-chat-completions builds it), the prompt hash
        and the cache key."""
        payload = self._create_payload(
            self.create_message(messages), generate=generate, gen_kwargs=copy.deepcopy(gen_kwargs),
            seed=self._seed, eos=self.eos_string, **kwargs,
        )
        first = messages[0] if isinstance(messages, (list, tuple)) and messages else None
        prompt_hash = hashlib.sha256(first.prompt.encode("utf-8")).hexdigest() if hasattr(first, "prompt") else None
        request_meta = {k: v for k, v in payload.items() if k != "messages"}
        request_meta["n_messages"] = len(payload.get("messages", []))
        key = cache_key(prompt_hash, request_meta) if prompt_hash else None
        return payload, prompt_hash, request_meta, key

    def is_cached(self, messages, gen_kwargs) -> bool:
        """Whether a complete answer to this exact request is already in the response cache."""
        key = self._prepare(messages, gen_kwargs)[3]
        return self.cache is not None and key is not None and self.cache.get(key) is not None

    def model_call(self, messages, *, generate=True, gen_kwargs=None, **kwargs):
        payload, prompt_hash, request_meta, key = self._prepare(messages, gen_kwargs, generate=generate, **kwargs)
        if self.cache is not None and key:  # not `if self.cache`: an empty cache has len() == 0
            cached = self.cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
        self._seq += 1

        for attempt in range(1, self.max_attempts + 1):
            self._pace(payload)
            started = time.time()
            record = {
                "session": self.session, "seq": self._seq, "attempt": attempt,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "model": self.model, "prompt_hash": prompt_hash, "request": request_meta,
            }
            try:
                resp = requests.post(self.base_url, json=payload, headers=self.header,
                                     timeout=self.timeout, verify=self.verify_certificate)
            except requests.RequestException as e:
                record.update(status=None, error=type(e).__name__, elapsed_s=round(time.time() - started, 3))
                self._log(record)
                if attempt == self.max_attempts:
                    raise GroqAPIError(f"network error after {attempt} attempts: {type(e).__name__}") from None
                time.sleep(min(60.0, 2.0 ** attempt))
                continue

            headers = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith(KEEP_HEADERS)}
            try:
                body = resp.json()
            except ValueError:
                body = {"_text": resp.text[:2000]}
            record.update(status=resp.status_code, elapsed_s=round(time.time() - started, 3),
                          headers=headers, response=body)
            self._log(record)
            self._note_headers(headers)

            if resp.status_code == 200:
                total = (body.get("usage") or {}).get("total_tokens") or 0
                self._window.append((time.time(), int(total)))
                self._recent_totals.append(int(total))
                if self.cache is not None and key:
                    self.cache.put(key, prompt_hash, self.model, body)  # refuses truncated/empty answers
                return body

            message = _error_message(body)
            if resp.status_code == 429:
                match = _TRY_AGAIN.search(message)
                wait = parse_duration(headers.get("retry-after")) or (parse_duration(match.group(1)) if match else None)
                daily = "per day" in message.lower() or (wait is not None and wait > self.daily_wait_limit_s)
                if daily:
                    raise QuotaExhausted(f"{self.model}: daily limit reached ({_ORG_ID.sub('org_REDACTED', message)[:300]})")
                if attempt == self.max_attempts:
                    raise GroqAPIError(f"{self.model}: still rate-limited after {attempt} attempts")
                time.sleep((wait if wait is not None else 5.0) + 0.5)
                continue
            if resp.status_code >= 500 and attempt < self.max_attempts:
                time.sleep(min(60.0, 2.0 ** attempt))
                continue
            raise GroqAPIError(f"{self.model}: HTTP {resp.status_code}: {_ORG_ID.sub('org_REDACTED', message)[:300]}")

        raise GroqAPIError(f"{self.model}: no response after {self.max_attempts} attempts")

    def parse_generations(self, outputs, **kwargs):
        """Content of each choice; None when truncated or empty (scored invalid, never cached)."""
        res = []
        for out in outputs if isinstance(outputs, list) else [outputs]:
            choices = (out or {}).get("choices") or []
            if not choices:
                res.append(None)
                continue
            texts = [None] * len(choices)
            for pos, choice in enumerate(choices):
                content = (choice.get("message") or {}).get("content")
                usable = choice.get("finish_reason") != "length" and content and content.strip()
                texts[choice.get("index", pos)] = (
                    postprocess_generated_text(content, stop=None, think_end_token=self.think_end_token)
                    if usable else None
                )
            res.extend(texts)
        return res
