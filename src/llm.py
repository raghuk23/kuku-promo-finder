"""LLM clients behind one tiny interface: complete_json(system, user, schema) -> dict.

Swapping the model -- including to whatever API Kuku provides in the interview --
is a config/env change, not a code change:
    LLM_PROVIDER=gemini|openai_compatible|heuristic   LLM_MODEL=<model id>
"""
import hashlib
import json
import os
import time
import urllib.request
from typing import Callable, Optional

from .config import ROOT

TRANSIENT_MARKERS = ("503", "429", "500", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                     "DEADLINE_EXCEEDED", "INTERNAL", "timed out", "Timeout",
                     "Connection reset", "temporarily")


class LLMError(Exception):
    pass


def is_transient(exc: Exception) -> bool:
    """Overload / rate-limit / network blips are worth retrying; 4xx are not."""
    s = f"{type(exc).__name__} {exc}"
    return any(m in s for m in TRANSIENT_MARKERS)


def with_retries(fn: Callable, retries: int = 4, base_delay: float = 2.0,
                 sleep: Callable = time.sleep):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- classified below
            if attempt == retries or not is_transient(e):
                raise
            sleep(base_delay * (2 ** attempt))


class JsonCache:
    """Content-addressed cache: same model + prompt -> same file. Re-runs are
    free and a run interrupted by quota limits resumes where it stopped."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    def get(self, key: str):
        f = os.path.join(self.path, key + ".json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                return json.load(fh)
        return None

    def set(self, key: str, value) -> None:
        f = os.path.join(self.path, key + ".json")
        tmp = f + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        os.replace(tmp, f)


class LLMClient:
    name = "base"
    last_model: Optional[str] = None

    def __init__(self):
        # Real token counts reported by the API, for the cost section of the doc.
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}

    def _add_usage(self, inp=0, out=0, think=0):
        if not hasattr(self, "usage"):
            LLMClient.__init__(self)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += inp or 0
        self.usage["output_tokens"] += out or 0
        self.usage["thinking_tokens"] += think or 0

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError


class _Paced:
    def __init__(self, min_interval_s: float):
        self.min_interval_s, self._last = min_interval_s, 0.0

    def wait(self):
        gap = time.monotonic() - self._last
        if gap < self.min_interval_s:
            time.sleep(self.min_interval_s - gap)
        self._last = time.monotonic()


def _loads(text: Optional[str]) -> dict:
    if not text:
        raise LLMError("empty response (possibly blocked by safety filters)")
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(t)


class GeminiClient(LLMClient):
    def __init__(self, model: str, fallback_models=(), temperature=0.0, seed=None,
                 retries=4, base_delay_s=4.0, min_interval_s=0.0, api_key=None):
        from google import genai
        super().__init__()
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key or key == "PASTE_YOUR_KEY_HERE":
            raise LLMError("GEMINI_API_KEY is not set (put it in .env)")
        self._genai_client = genai.Client(api_key=key)
        self.models = [model] + [m for m in fallback_models if m != model]
        self.name = f"gemini:{model}"
        self.temperature, self.seed = temperature, seed
        self.retries, self.base_delay_s = retries, base_delay_s
        self._pace = _Paced(min_interval_s)

    def _call(self, model, system, user, schema):
        from google.genai import types
        self._pace.wait()
        r = self._genai_client.models.generate_content(
            model=model, contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=self.temperature, seed=self.seed,
                response_mime_type="application/json", response_json_schema=schema))
        u = r.usage_metadata
        if u is not None:
            self._add_usage(getattr(u, "prompt_token_count", 0), getattr(u, "candidates_token_count", 0),
                            getattr(u, "thoughts_token_count", 0))
        return _loads(r.text)

    def complete_json(self, system, user, schema):
        last_err = None
        for m in self.models:
            try:
                out = with_retries(lambda: self._call(m, system, user, schema),
                                   self.retries, self.base_delay_s)
                self.last_model = m
                return out
            except Exception as e:  # noqa: BLE001 -- try the next model, keep the error
                last_err = e
        raise last_err


class OpenAICompatibleClient(LLMClient):
    """Any /chat/completions endpoint: OpenRouter, Groq, Together, Anthropic's
    OpenAI-compatible endpoint, a local server..."""

    def __init__(self, model, base_url, api_key_env, temperature=0.0, retries=4,
                 base_delay_s=4.0, min_interval_s=0.0):
        super().__init__()
        self.key = os.environ.get(api_key_env)
        if not self.key:
            raise LLMError(f"{api_key_env} is not set")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.name = self.last_model = f"openai_compatible:{model}"
        self.temperature, self.retries, self.base_delay_s = temperature, retries, base_delay_s
        self._pace = _Paced(min_interval_s)

    def _call(self, system, user, schema):
        self._pace.wait()
        body = json.dumps({
            "model": self.model, "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system + "\n\nReturn only JSON matching this schema:\n"
                 + json.dumps(schema)},
                {"role": "user", "content": user}],
        }).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)
        u = data.get("usage") or {}
        self._add_usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        return _loads(data["choices"][0]["message"]["content"])

    def complete_json(self, system, user, schema):
        return with_retries(lambda: self._call(system, user, schema), self.retries, self.base_delay_s)


def make_client(cfg: dict) -> Optional[LLMClient]:
    """Build the configured client, or None for the no-LLM heuristic judge."""
    llm = cfg["llm"]
    provider = os.environ.get("LLM_PROVIDER", llm.get("provider", "heuristic"))
    model = os.environ.get("LLM_MODEL", llm.get("model"))
    if provider == "heuristic":
        return None
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
    common = dict(temperature=llm.get("temperature", 0), retries=llm.get("retries", 4),
                  base_delay_s=llm.get("base_delay_s", 4), min_interval_s=llm.get("min_interval_s", 0))
    if provider == "gemini":
        fallbacks = [] if "LLM_MODEL" in os.environ else llm.get("fallback_models", [])
        return GeminiClient(model, fallbacks, seed=llm.get("seed"), **common)
    if provider == "openai_compatible":
        oc = llm["openai_compatible"]
        return OpenAICompatibleClient(model, oc["base_url"], oc["api_key_env"], **common)
    raise LLMError(f"unknown provider {provider!r}")
