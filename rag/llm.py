"""The chat model, set up to stay on free tiers.

Models are tried in order with LangChain's `with_fallbacks`: when one returns a
rate-limit or quota error (or any error), the next one answers. Retries are off
on purpose, so a 429 moves on at once instead of waiting.

With only GEMINI_API_KEY set it uses Google AI Studio's free tier through the
OpenAI-compatible endpoint:  gemini-flash-latest -> gemini-flash-lite-latest -> gemini-3.1-flash-lite.
A per-instance daily cap keeps a public demo inside that quota. Past the cap, or
if every model fails, the pipelines fall back to extractive answers instead of
showing an error.
"""
import os
import threading
import time

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# "-latest" aliases follow Google's current Flash models, so a retired version doesn't break the bot
# (gemini-2.5-flash already returns 404). All three have free-tier quota in AI Studio.
FREE_GEMINI_MODELS = "gemini-flash-latest,gemini-flash-lite-latest,gemini-3.1-flash-lite"


class LLMUnavailable(Exception):
    pass


class FallbackLLM:
    def __init__(self, key: str, base_url: str | None, models: list[str], effort: str | None, effort_if: str = ""):
        from langchain_openai import ChatOpenAI

        def one(model: str):
            kw = dict(model=model, api_key=key, temperature=0, timeout=float(os.getenv("LLM_TIMEOUT", "25")), max_retries=0)
            if base_url:
                kw["base_url"] = base_url
            if effort and effort_if in model:
                kw["reasoning_effort"] = effort
            return ChatOpenAI(**kw)

        self.models = models
        runs = [one(m) for m in models]
        self.runnable = runs[0].with_fallbacks(runs[1:]) if len(runs) > 1 else runs[0]
        self.cap = int(os.getenv("DAILY_LLM_CAP", "400"))
        self.day, self.used = "", 0
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.models[0] if len(self.models) == 1 else f"{self.models[0]} (+{len(self.models) - 1} fallbacks)"

    def invoke(self, prompt, inputs: dict) -> tuple[str, str]:
        """Returns (text, model that answered). Raises LLMUnavailable when the cap is hit or every model fails."""
        with self._lock:
            today = time.strftime("%Y-%m-%d", time.gmtime())
            if today != self.day:
                self.day, self.used = today, 0
            if self.used >= self.cap:
                raise LLMUnavailable(f"the demo's daily cap of {self.cap} model calls is used up")
            self.used += 1
        try:
            msg = (prompt | self.runnable).invoke(inputs)
        except Exception as e:  # every model in the chain failed
            raise LLMUnavailable(f"all free-tier models refused ({type(e).__name__})") from e
        text = msg.content if isinstance(msg.content, str) else "".join(p.get("text", "") for p in msg.content if isinstance(p, dict))
        return text.strip(), msg.response_metadata.get("model_name") or self.models[0]


def make_llm() -> FallbackLLM | None:
    split = lambda s: [m.strip() for m in s.split(",") if m.strip()]  # noqa: E731
    if os.getenv("OPENAI_API_KEY"):
        return FallbackLLM(os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_BASE_URL") or None,
                           split(os.getenv("OPENAI_MODEL", "gpt-4o-mini")), os.getenv("OPENAI_REASONING_EFFORT"))
    if os.getenv("GEMINI_API_KEY"):
        # Low thinking: fast, and plenty for answering from five excerpts ("none" is rejected by the lite models).
        return FallbackLLM(os.environ["GEMINI_API_KEY"], GEMINI_URL, split(os.getenv("GEMINI_MODELS", FREE_GEMINI_MODELS)),
                           os.getenv("GEMINI_REASONING_EFFORT", "low"))
    return None
