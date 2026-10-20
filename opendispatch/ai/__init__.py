"""AI helpers for Open-Dispatch.

Currently provides per-platform caption adaptation via OpenRouter or local
Ollama. Designed so the user can swap providers without changing callers.
"""

from .caption_adapter import adapt_caption, adapt_caption_async, AdaptError

__all__ = ["adapt_caption", "adapt_caption_async", "AdaptError"]
