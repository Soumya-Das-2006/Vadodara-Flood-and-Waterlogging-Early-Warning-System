"""Single entry point for every LLM task (alerts, RAG chat). Default provider: Groq (OpenAI-compatible API).
Keys come from environment variables only (never hard-coded):
  GROQ_API_KEY   (required for Groq)      GROQ_MODEL (default llama-3.3-70b-versatile)
  LLM_PROVIDER   groq (default) | anthropic   (anthropic uses ANTHROPIC_API_KEY / CLAUDE_MODEL)
Without a key, callers fall back to validated templates / rule-based answers."""
import os
import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def provider():
    return os.environ.get("LLM_PROVIDER", "groq").lower()


def available():
    return bool(os.environ.get("GROQ_API_KEY")) if provider() == "groq" else bool(os.environ.get("ANTHROPIC_API_KEY"))


def model_name():
    if provider() == "groq":
        return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    return os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")


def chat(system, user, max_tokens=800, json_mode=False, timeout=20.0):
    """Returns the model's text. Raises on any failure (callers catch and fall back)."""
    if provider() == "groq":
        body = {"model": model_name(), "temperature": 0.2, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = requests.post(GROQ_URL, json=body, timeout=timeout,
                          headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    import anthropic
    c = anthropic.Anthropic(timeout=timeout, max_retries=1)
    m = c.messages.create(model=model_name(), max_tokens=max_tokens, system=system,
                          messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
