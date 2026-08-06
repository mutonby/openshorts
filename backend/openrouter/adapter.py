from typing import List, Dict, Any, Optional
import asyncio
import httpx
import time

from .manager import get_manager
from .config import OPENROUTER_RETRY_ATTEMPTS

OPENROUTER_CHAT_ENDPOINT = "https://api.openrouter.ai/v1/chat/completions"


async def send_chat_completion(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
    """Send a chat completion request via OpenRouter using a selected model from the rotator.

    messages should be a list of dicts: [{"role": "user", "content": "..."}, ...]
    Returns the raw response JSON from OpenRouter.
    """
    mgr = get_manager()

    last_err = None
    attempts = 0
    while attempts < OPENROUTER_RETRY_ATTEMPTS:
        attempts += 1
        model = await mgr.get_model()
        if not model:
            raise RuntimeError("No OpenRouter models available in roster. Set OPENROUTER_API_KEY and try again.")

        model_id = model.get("id")
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {"Authorization": f"Bearer {mgr._api_key}", "Content-Type": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(OPENROUTER_CHAT_ENDPOINT, json=payload, headers=headers)
                # If rate-limited or server error, mark model unhealthy and retry
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    await mgr.mark_unhealthy(model_id)
                    last_err = RuntimeError(f"OpenRouter model {model_id} returned status {resp.status_code}")
                    await asyncio.sleep(min(2 ** attempts, 10))
                    continue

                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            # mark model unhealthy and try next
            await mgr.mark_unhealthy(model_id)
            last_err = e
            await asyncio.sleep(min(2 ** attempts, 10))
            continue

    # If we get here, all attempts failed
    raise last_err or RuntimeError("OpenRouter request failed after retries")
