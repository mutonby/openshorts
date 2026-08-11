"""ai_gateway.py — Zero-Budget AI Gateway for OpenShorts+

Routes every AI call in the pipeline through OpenAI-compatible endpoints so the
whole app can run on FREE tiers. Supported out of the box:

  * OpenRouter    (OPENROUTER_API_KEY) — dozens of ``:free`` models
    (DeepSeek, Qwen, GLM, Kimi, Gemma, gpt-oss, Nemotron, ...)
  * DeepSeek      (DEEPSEEK_API_KEY)   — deepseek-chat / deepseek-reasoner
  * Moonshot      (MOONSHOT_API_KEY)   — Kimi models (moonshot-v1, kimi-k2, ...)
  * Zhipu / GLM   (ZHIPU_API_KEY)      — glm-4.5-air, glm-4-flash (free tier)
  * Alibaba Qwen  (DASHSCOPE_API_KEY)  — qwen-plus / qwen-turbo (DashScope
    compatible-mode; new accounts get free quota)
  * Groq          (GROQ_API_KEY)       — llama-3.3-70b-versatile, llama-3.1-8b
    (generous free tier)
  * Google AI     (GEMINI_API_KEY)     — Gemini free tier via Google's
    OpenAI-compatible endpoint
  * Any OpenAI-compatible base URL     (OPENAI_API_BASE / OPENAI_API_KEY)
    — AgentRouter, token routers, LiteLLM proxies, local vLLM/llama.cpp, ...

How it works
------------
A *chain* is an ordered list of ``provider:model`` entries. ``complete()``
walks the chain: on rate-limit, 5xx, timeout, or an unparsable JSON answer it
falls through to the next entry, so one provider burning through its free quota
never blocks the pipeline. Providers without an API key are skipped entirely.

The chain is configurable with ``AI_MODEL_CHAIN`` (text), ``AI_VISION_MODEL_CHAIN``
(vision), ``AI_IMAGE_MODEL`` (image generation). Defaults point at today's free
models; the roster rotates, so treat the defaults as suggestions and override
in ``.env`` when a model is retired.

Everything is metered as $0.00 — these are free tiers — but token usage is
tracked so the UI can still show what the pipeline did.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

# Sentinel returned by app.resolve_gemini() when the gateway handles providers
# directly (no single "Gemini key" anymore).
GATEWAY_SENTINEL = "__gateway__"

# --------------------------------------------------------------------------- #
# Provider registry
# --------------------------------------------------------------------------- #
# Each provider is an OpenAI-compatible chat/completions endpoint. ``key_env``
# names the env var holding the API key; the provider is inactive without it.
PROVIDERS: Dict[str, Dict[str, Any]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "default_models": ["openai/gpt-oss-20b:free"],
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "key_env": "DEEPSEEK_API_KEY",
        "default_models": ["deepseek-chat"],
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "key_env": "MOONSHOT_API_KEY",
        "default_models": ["moonshot-v1-8k"],
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "key_env": "ZHIPU_API_KEY",
        "default_models": ["glm-4.5-air"],
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "key_env": "DASHSCOPE_API_KEY",
        "default_models": ["qwen-plus"],
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "default_models": ["llama-3.3-70b-versatile"],
    },
    "google": {
        # Google AI Studio's OpenAI-compatible endpoint. GEMINI_API_KEY keeps
        # its name for backward compatibility with existing .env files.
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "key_env": "GEMINI_API_KEY",
        "default_models": ["gemini-2.5-flash"],
    },
    "openai": {
        # Generic OpenAI-compatible endpoint (AgentRouter, token routers,
        # LiteLLM proxies, local servers...). Base URL and model are read from
        # OPENAI_API_BASE / OPENAI_API_MODEL so any free router plugs in.
        "base_url_env": "OPENAI_API_BASE",
        "key_env": "OPENAI_API_KEY",
        "default_models": ["openai-api-model"],  # overridden by OPENAI_API_MODEL
    },
}

# Default text chain (August 2026 free roster — override with AI_MODEL_CHAIN).
# Entries are tried in order; only providers with a configured key are used.
DEFAULT_TEXT_CHAIN = (
    "openrouter:openai/gpt-oss-20b:free,"
    "openrouter:google/gemma-4-31b-it:free,"
    "openrouter:qwen/qwen3-32b:free,"
    "openrouter:z-ai/glm-4.5-air:free,"
    "openrouter:deepseek/deepseek-r1-distill:free,"
    "groq:llama-3.3-70b-versatile,"
    "zhipu:glm-4.5-air,"
    "deepseek:deepseek-chat,"
    "dashscope:qwen-plus,"
    "moonshot:moonshot-v1-8k,"
    "google:gemini-2.5-flash"
)

# Default vision chain (multimodal free models that accept image inputs).
DEFAULT_VISION_CHAIN = (
    "openrouter:google/gemma-4-31b-it:free,"
    "openrouter:nvidia/nemotron-nano-12b-v2-vl:free,"
    "google:gemini-2.5-flash,"
    "openrouter:qwen/qwen3-vl-plus:free"
)

# Default image-generation model (OpenRouter images endpoint). Rotates often;
# if unset or unavailable the caller falls back to local generation.
DEFAULT_IMAGE_MODEL = "openrouter:google/gemini-2.5-flash-image"

DEFAULT_MAX_TOKENS = 4096

# HTTP statuses worth retrying on the same provider.
_TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}

_TIMEOUT = httpx.Timeout(30.0, read=300.0)

# OpenRouter free-model autodiscovery. The API returns every model with its
# pricing; free models have $0 pricing (or a ":free" suffix). We cache the
# list for a few hours and build the default chain from it, so when a model is
# retired the next refresh simply routes around it — no config edits.
_FREE_MODELS_CACHE: Dict[str, Any] = {"at": 0.0, "text": [], "vision": []}
_FREE_MODELS_TTL = 6 * 3600
FREE_MODEL_AUTODISCOVERY = os.environ.get(
    "FREE_MODEL_AUTODISCOVERY", "1").strip().lower() in ("1", "true", "yes")

# Provider failure memory: when a provider answers 429/5xx/timeout we put it
# in cooldown and the chain skips it for a while, so a limited model never
# blocks the pipeline and traffic shifts to healthy providers automatically.
_FAILURES: Dict[str, Dict[str, float]] = {}
_COOLDOWN_BASE_SECONDS = 45
_COOLDOWN_MAX_SECONDS = 10 * 60


class AIGatewayError(RuntimeError):
    """All providers in the chain failed for the request."""


@dataclass
class AIResult:
    """One completed AI call: the answer plus what served it."""

    text: str = ""
    model: str = ""
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # free tiers: always 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def usage_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.cost,
            "model": self.model,
            "provider": self.provider,
            "price_estimated": False,
        }


# --------------------------------------------------------------------------- #
# Chain resolution
# --------------------------------------------------------------------------- #
def _provider_base(provider: str) -> Optional[str]:
    spec = PROVIDERS.get(provider)
    if not spec:
        return None
    if "base_url_env" in spec:
        return os.environ.get(spec["base_url_env"], "").strip() or None
    return spec["base_url"]


def provider_key(provider: str) -> Optional[str]:
    """The primary API key for a provider, or None when not configured."""
    keys = provider_keys(provider)
    return keys[0] if keys else None


def provider_keys(provider: str) -> List[str]:
    """Every API key configured for a provider.

    The first key comes from the provider's KEY env var; additional keys use
    the numbered scheme KEY_2, KEY_3... so the settings UI can store as many
    free keys per provider as the user likes.
    """
    spec = PROVIDERS.get(provider)
    if not spec:
        return []
    keys: List[str] = []
    env = spec["key_env"]
    primary = os.environ.get(env, "").strip()
    if primary:
        keys.append(primary)
    i = 2
    while True:
        extra = os.environ.get(f"{env}_{i}", "").strip()
        if not extra:
            break
        keys.append(extra)
        i += 1
    return keys


def _split_entry(entry: str) -> Tuple[str, str]:
    entry = (entry or "").strip()
    if not entry:
        return "", ""
    if ":" in entry:
        provider, model = entry.split(":", 1)
        return provider.strip().lower(), model.strip()
    # Bare model: use the first configured provider.
    for p in PROVIDERS:
        if provider_key(p):
            return p, entry.strip()
    return "openrouter", entry.strip()


def resolve_chain(kind: str = "text") -> List[Tuple[str, str, str]]:
    """The ordered, configured (provider, model, key) chain for a task kind.

    kind: "text" | "vision" | "image". Entries whose provider has no key are
    dropped; every configured key of a provider is expanded into its own
    chain entry. Returns [] when nothing is configured at all.

    With an OpenRouter key and no explicit AI_MODEL_CHAIN, the chain is built
    from OpenRouter's catalog filtered to FREE models only (never paid), with
    the other configured providers' defaults appended as extra fallbacks —
    so the app auto-adapts when free models are added/retired.
    """
    explicit = os.environ.get("AI_MODEL_CHAIN") or os.environ.get(
        "FREE_AI_MODEL_CHAIN")
    vision_chain = os.environ.get("AI_VISION_MODEL_CHAIN")
    image_model = os.environ.get("AI_IMAGE_MODEL")

    entries: List[Tuple[str, str]] = []
    if kind == "vision" and vision_chain:
        for entry in vision_chain.split(","):
            entries.append(_split_entry(entry))
    elif kind == "image" and image_model:
        entries.append(_split_entry(image_model))
    elif kind == "text" and explicit:
        for entry in explicit.split(","):
            entries.append(_split_entry(entry))
    else:
        # Auto-discovery: free OpenRouter models first, then configured
        # providers' defaults as fallback.
        openrouter_discovered = False
        if FREE_MODEL_AUTODISCOVERY and provider_key("openrouter"):
            free = discover_openrouter_free_models(
                kind=kind, limit=12 if kind == "text" else 6)
            if free:
                openrouter_discovered = True
                for mid in free:
                    entries.append(("openrouter", mid))
        default_raw = (DEFAULT_VISION_CHAIN if kind == "vision"
                       else DEFAULT_IMAGE_MODEL if kind == "image"
                       else DEFAULT_TEXT_CHAIN)
        for entry in default_raw.split(","):
            provider, model = _split_entry(entry)
            if provider == "openrouter" and openrouter_discovered:
                continue  # already covered by discovered free models
            entries.append((provider, model))

    # Legacy GEMINI_MODEL env vars keep working: a named Gemini model is
    # prepended to whatever chain is active (only usable when a key exists).
    legacy_model = os.environ.get("GEMINI_MODEL", "").strip()
    if not explicit and kind == "text" and legacy_model:
        entries.insert(0, ("google", legacy_model))

    chain: List[Tuple[str, str, str]] = []
    for provider, model in entries:
        if not provider or not model:
            continue
        if not _provider_base(provider):
            continue
        for key in provider_keys(provider):
            chain.append((provider, model, key))
    return chain


def is_configured() -> bool:
    """True when at least one AI provider has a key set."""
    return any(provider_key(p) for p in PROVIDERS)


def configured_providers() -> List[str]:
    return [p for p in PROVIDERS if provider_key(p)]


# --------------------------------------------------------------------------- #
# Failure memory — automatic switching when a provider gets limited
# --------------------------------------------------------------------------- #
def _cooldown_seconds(provider: str) -> float:
    state = _FAILURES.get(provider)
    if not state:
        return 0.0
    return max(0.0, state.get("until", 0.0) - time.time())


def provider_in_cooldown(provider: str) -> bool:
    return _cooldown_seconds(provider) > 0


def _mark_failure(provider: str) -> None:
    state = _FAILURES.setdefault(provider, {"count": 0, "until": 0.0})
    state["count"] = int(state.get("count", 0)) + 1
    delay = min(_COOLDOWN_MAX_SECONDS,
                _COOLDOWN_BASE_SECONDS * (2 ** (state["count"] - 1)))
    state["until"] = time.time() + delay
    print(f"⚠️ [ai_gateway] {provider} limited — skipping it for "
          f"{int(delay)}s (failures={state['count']})")


def _mark_success(provider: str) -> None:
    if provider in _FAILURES:
        state = _FAILURES[provider]
        state["count"] = max(0, int(state.get("count", 0)) - 1)
        if state["count"] == 0:
            _FAILURES.pop(provider, None)


def provider_status() -> Dict[str, Dict[str, Any]]:
    """Health snapshot for the UI: cooldown + failure counts per provider."""
    out = {}
    for provider in PROVIDERS:
        keys = provider_keys(provider)
        if not keys:
            continue
        state = _FAILURES.get(provider)
        out[provider] = {
            "keys": len(keys),
            "cooldown": round(_cooldown_seconds(provider), 1),
            "failures": int((state or {}).get("count", 0)),
        }
    return out


# --------------------------------------------------------------------------- #
# OpenRouter free-model autodiscovery (only free models — never paid)
# --------------------------------------------------------------------------- #
def discover_openrouter_free_models(key: Optional[str] = None,
                                     kind: str = "text",
                                     limit: int = 10) -> List[str]:
    """Fetch OpenRouter's catalog and return ONLY free model ids.

    Filters out every paid model: a model is free when its prompt and
    completion prices are both $0, or its id ends with ':free'. Results are
    ordered by OpenRouter's own popularity rank. Cached for _FREE_MODELS_TTL.
    Vision results only include models that accept image input.

    Returns [] when the fetch fails — callers then use the built-in defaults.
    """
    global _FREE_MODELS_CACHE
    now = time.time()
    cached = _FREE_MODELS_CACHE
    if now - cached.get("at", 0) < _FREE_MODELS_TTL and cached.get(kind):
        return cached[kind][:limit]

    key = key or provider_key("openrouter")
    if not key:
        return []
    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(20.0),
        )
        resp.raise_for_status()
        models = resp.json().get("data") or []
    except Exception as e:
        print(f"⚠️ [ai_gateway] OpenRouter catalog fetch failed: {e}")
        return []

    free_text, free_vision = [], []
    for m in models:
        mid = str(m.get("id") or "")
        pricing = m.get("pricing") or {}

        def _zero(v):
            try:
                return float(v) == 0.0
            except (TypeError, ValueError):
                return False

        is_free = mid.endswith(":free") or (
            _zero(pricing.get("prompt")) and _zero(pricing.get("completion")))
        if not is_free:
            continue  # paid models are never used
        arch = m.get("architecture") or {}
        modalities = [str(x).lower() for x in arch.get("input_modalities", [])]
        is_vision = any(mm in modalities for mm in ("image", "video"))
        entry = (int(m.get("order") or 999999), mid)
        (free_vision if is_vision else free_text).append(entry)

    free_text.sort(key=lambda e: e[0])
    free_vision.sort(key=lambda e: e[0])
    _FREE_MODELS_CACHE = {
        "at": now,
        "text": [mid for _o, mid in free_text],
        "vision": [mid for _o, mid in free_vision],
    }
    print(f"✅ [ai_gateway] Discovered {len(free_text)} free text + "
          f"{len(free_vision)} free vision models on OpenRouter")
    return (_FREE_MODELS_CACHE["vision"] if kind == "vision"
            else _FREE_MODELS_CACHE["text"])[:limit]


def _expand_chain(entries: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """Expand (provider, model) entries across every configured key of the
    provider, so multiple keys per provider all get used (a key's provider is
    skipped entirely while it is in cooldown)."""
    chain: List[Tuple[str, str, str]] = []
    for provider, model in entries:
        for key in provider_keys(provider):
            chain.append((provider, model, key))
    return chain


# --------------------------------------------------------------------------- #
# Request building
# --------------------------------------------------------------------------- #
def _build_messages(
    system: Optional[str],
    user: Optional[str],
    messages: Optional[List[Dict[str, Any]]],
    images: Optional[Sequence[bytes]],
) -> List[Dict[str, Any]]:
    """Build OpenAI-format messages, inlining images as base64 data URLs."""
    if messages is not None:
        return list(messages)

    parts: List[Dict[str, Any]] = []
    if images:
        for img in images:
            b64 = base64.b64encode(bytes(img)).decode("ascii")
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
    if user:
        parts.append({"type": "text", "text": user})

    out: List[Dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    if parts:
        out.append({"role": "user", "content": parts})
    return out


def _parse_json_text(text: str) -> Optional[Any]:
    """Best-effort JSON extraction from a model answer (fences included).

    Accepts either a JSON object or a JSON array at the top level.
    """
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    opens = [i for i, ch in enumerate(cleaned) if ch in "[{"]
    closes = [i for i, ch in enumerate(cleaned) if ch in "]}"][::-1]
    if opens and closes and closes[0] > opens[0]:
        start, end = opens[0], closes[0]
        cleaned = cleaned[start:end + 1]
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, (dict, list)) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Core completion
# --------------------------------------------------------------------------- #
def _chat_completion(
    provider: str,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    timeout: httpx.Timeout,
) -> AIResult:
    """One chat completion against one provider. Raises on failure."""
    base = _provider_base(provider)
    assert base, f"provider {provider} has no base URL"
    url = base.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/tahsinxiao/openshort-"
        headers["X-Title"] = "OpenShorts+ (zero-budget edition)"

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    if provider == "openrouter":
        body["route"] = "fallback"  # OpenRouter itself retries the model

    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=headers, json=body)
        if resp.status_code in _TRANSIENT_STATUS:
            resp.raise_for_status()  # caller retries on transient
        if resp.status_code >= 400:
            # Some models reject response_format — retry without JSON mode.
            if json_mode and resp.status_code in (400, 404, 422):
                body.pop("response_format", None)
                resp = client.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    resp.raise_for_status()
            else:
                resp.raise_for_status()
        data = resp.json()

    text = ""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):  # some providers return parts
            text = "".join(
                p.get("text", "") for p in content if isinstance(p, dict))
        else:
            text = str(content or "")
    usage = data.get("usage") or {}
    return AIResult(
        text=text.strip(),
        model=data.get("model") or model,
        provider=provider,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        raw=data,
    )


def complete(
    system: Optional[str] = None,
    user: Optional[str] = None,
    *,
    messages: Optional[List[Dict[str, Any]]] = None,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_mode: bool = False,
    images: Optional[Sequence[bytes]] = None,
    kind: str = "text",
    chain: Optional[List[Tuple[str, str, str]]] = None,
    timeout: httpx.Timeout = _TIMEOUT,
) -> AIResult:
    """Walk the chain until one provider answers. Raises AIGatewayError."""
    chain = chain if chain is not None else resolve_chain(kind)
    if not chain:
        raise AIGatewayError(
            "No free AI provider configured. Set OPENROUTER_API_KEY, "
            "GEMINI_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY, ZHIPU_API_KEY, "
            "DASHSCOPE_API_KEY or MOONSHOT_API_KEY in your .env "
            "(see .env.example). All of them have free tiers — zero budget.")
    if not system and not user and messages is None:
        raise AIGatewayError("complete() needs system/user text or messages")

    msgs = _build_messages(system, user, messages, images)
    last_error: Optional[Exception] = None
    tried = 0

    for provider, model, api_key in chain:
        if provider_in_cooldown(provider):
            continue  # auto-switch: skip providers that were just limited
        tried += 1
        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                result = _chat_completion(
                    provider, model, api_key, msgs,
                    temperature=temperature, max_tokens=max_tokens,
                    json_mode=json_mode, timeout=timeout,
                )
                if json_mode and not _parse_json_text(result.text):
                    raise ValueError("JSON mode: answer did not parse as JSON")
                _mark_success(provider)
                return result
            except Exception as e:  # noqa: BLE001 - fall through the chain
                last_error = e
                msg = str(e)
                transient = any(tok in msg for tok in (
                    "429", "500", "502", "503", "504", "timeout", "Timed out",
                    "Connection", "ReadTimeout", "ConnectTimeout",
                    "did not parse as JSON", "ServiceUnavailable",
                    "Overloaded", "overloaded",
                ))
                if transient:
                    _mark_failure(provider)
                    if attempt < attempts:
                        time.sleep(2.0 * attempt)
                        continue
                break  # next provider

    raise AIGatewayError(
        f"All {len(chain)} AI provider(s) failed"
        f" ({tried} tried). Last error: {last_error}")


def complete_json(
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    images: Optional[Sequence[bytes]] = None,
    kind: str = "text",
    chain: Optional[List[Tuple[str, str, str]]] = None,
    max_attempts: int = 2,
) -> Tuple[Any, AIResult]:
    """Ask the chain for JSON (object or array), return (parsed, result)."""
    last_error: Optional[Exception] = None
    for _ in range(max_attempts):
        try:
            result = complete(
                system=system, user=user, temperature=temperature,
                max_tokens=max_tokens, json_mode=True, images=images,
                kind=kind, chain=chain,
            )
            parsed = _parse_json_text(result.text)
            if parsed is not None:
                return parsed, result
            last_error = ValueError("AI returned unparsable JSON")
        except AIGatewayError as e:
            last_error = e
        except Exception as e:  # noqa: BLE001
            last_error = e
    raise AIGatewayError(
        f"AI JSON generation failed after {max_attempts} attempt(s): "
        f"{last_error}")


def chat(
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.7,
    kind: str = "text",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> AIResult:
    """Free-form conversation (e.g. the title refinement chat)."""
    return complete(messages=messages, temperature=temperature,
                    kind=kind, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# Frame sampling (vision tasks without uploading the whole video)
# --------------------------------------------------------------------------- #
def _probe_duration(video_path: str) -> float:
    """Video duration in seconds via ffprobe, falling back to ffmpeg -i."""
    try:
        probe = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            stderr=subprocess.STDOUT, timeout=60,
        ).decode().strip()
        return max(0.0, float(probe))
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", video_path],
            capture_output=True, timeout=60)
        match = re.search(
            r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)",
            result.stderr.decode(errors="ignore"))
        if match:
            h, mnt, sec = match.groups()
            return int(h) * 3600 + int(mnt) * 60 + float(sec)
    except Exception:
        pass
    return 0.0


def extract_frames(
    video_path: str,
    n: int = 12,
    width: int = 1024,
) -> List[bytes]:
    """Evenly-spaced JPEG frames from a video, via ffmpeg.

    Free vision models get a handful of stills instead of a multi-GB upload —
    same technique the layout picker already validated (see layout_picker.py).
    Returns [] on any failure; callers treat that as "no vision available".
    """
    duration = _probe_duration(video_path)
    if duration <= 0:
        return []

    n = max(2, min(int(n), 30))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            pattern = os.path.join(tmp, "frame_%03d.jpg")
            cmd = [
                "ffmpeg", "-y", "-v", "error", "-i", video_path,
                "-vf", f"fps={n}/{duration:.6f},scale={width}:-2",
                "-frames:v", str(n), "-q:v", "5", pattern,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=300)
            frames = []
            for i in range(1, n + 1):
                p = os.path.join(tmp, f"frame_{i:03d}.jpg")
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    with open(p, "rb") as f:
                        frames.append(f.read())
            return frames
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Image generation (optional — falls back to local rendering when unavailable)
# --------------------------------------------------------------------------- #
def generate_image(
    prompt: str,
    output_path: str,
    *,
    size: str = "1024x1024",
    timeout: httpx.Timeout = httpx.Timeout(60.0, read=240.0),
) -> Optional[str]:
    """Generate an image with the configured free image model.

    Uses the OpenAI-compatible /images/generations endpoint (OpenRouter and
    friends). Returns output_path on success, None when no image provider is
    configured or the call failed — callers then use their local fallback.
    """
    chain = resolve_chain("image")
    if not chain:
        return None
    provider, model, api_key = chain[0]
    base = _provider_base(provider)
    if not base:
        return None
    url = base.rstrip("/") + "/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {"model": model, "prompt": prompt, "size": size, "n": 1}
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
        item = (data.get("data") or [{}])[0]
        b64 = item.get("b64_json")
        if b64:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(b64))
            return output_path
        url_out = item.get("url")
        if url_out:
            with httpx.Client(timeout=timeout) as client:
                img = client.get(url_out)
                img.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(img.content)
            return output_path
    except Exception as e:
        print(f"⚠️ [ai_gateway] Image generation failed ({model}): {e}")
        return None
    return None


# --------------------------------------------------------------------------- #
# Gemini image generation (legacy BYOK path — only when a real Gemini key is
# set, i.e. not the gateway sentinel). Used by thumbnail.py.
# --------------------------------------------------------------------------- #
def gemini_image_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key or key == GATEWAY_SENTINEL:
        return None
    return key
