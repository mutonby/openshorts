"""
Multi-provider LLM adapter with configurable fallback chain.

Supports Gemini, Claude (Anthropic API + Bedrock), OpenAI, Amazon Bedrock (Nova),
Grok, and DeepSeek. Each provider implements the same schema (ScoreResponse,
DetailResponse) so the rest of the pipeline sees a uniform interface.

Configuration:
  - LLM_FALLBACK_ORDER: comma-separated provider names (default: gemini,claude,deepseek,openai,grok)
  - Provider-specific API keys via env vars: GEMINI_API_KEY, ANTHROPIC_API_KEY,
    OPENAI_API_KEY, DEEPSEEK_API_KEY, XAI_API_KEY, AWS_REGION (for Bedrock)
  - Log which provider ultimately handled each job via logging.info()
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Transient or fatal error from a provider."""
    pass


class BaseProvider(ABC):
    """Common interface for clip-selection providers."""

    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g., 'gemini', 'claude', 'openai')."""
        pass

    @abstractmethod
    def available(self) -> bool:
        """True if this provider is ready to use (API key, network, etc.)."""
        pass

    @abstractmethod
    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Score windows using this provider.
        Returns (parsed_response, cost_analysis).
        parsed_response should have a 'windows' key with scored windows.
        cost_analysis should have 'input_tokens', 'output_tokens', 'total_cost', 'model'.
        """
        pass

    @abstractmethod
    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Extract detailed clips using this provider.
        Returns (parsed_response, cost_analysis).
        parsed_response should have a 'shorts' key with clip details.
        cost_analysis should have 'input_tokens', 'output_tokens', 'total_cost', 'model'.
        """
        pass


class GeminiProvider(BaseProvider):
    """Google Gemini via google-ai Python SDK."""

    def name(self) -> str:
        return "gemini"

    def available(self) -> bool:
        return bool(os.getenv("GEMINI_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Gemini."""
        try:
            from google import genai
            from google.genai import types as genai_types
            import gemini_worker
        except ImportError:
            raise ProviderError("Gemini SDK not available; install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.ScoreResponse,
        )

        response = client.models.generate_content(
            model=model or "gemini-3.1-flash-lite",
            contents=prompt,
            config=config
        )
        gemini_worker.raise_if_blocked(response)

        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        else:
            parsed = gemini_worker._parse_json_response_text(
                gemini_worker._get_response_text(response)
            )
        cost = gemini_worker._calculate_cost_analysis(response, model or "gemini-3.1-flash-lite")
        return parsed, cost or {}

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Gemini."""
        try:
            from google import genai
            from google.genai import types as genai_types
            import gemini_worker
        except ImportError:
            raise ProviderError("Gemini SDK not available")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ProviderError("GEMINI_API_KEY not set")

        client = genai.Client(api_key=api_key)
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=gemini_worker.DetailResponse,
        )

        response = client.models.generate_content(
            model=model or "gemini-3.1-flash-lite",
            contents=prompt,
            config=config
        )
        gemini_worker.raise_if_blocked(response)

        parsed_obj = getattr(response, "parsed", None)
        if parsed_obj is not None:
            parsed = parsed_obj.model_dump() if hasattr(parsed_obj, "model_dump") else parsed_obj
        else:
            parsed = gemini_worker._parse_json_response_text(
                gemini_worker._get_response_text(response)
            )
        cost = gemini_worker._calculate_cost_analysis(response, model or "gemini-3.1-flash-lite")
        return parsed, cost or {}


class ClaudeProvider(BaseProvider):
    """Claude via Anthropic API."""

    def name(self) -> str:
        return "claude"

    def available(self) -> bool:
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Claude."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ProviderError("Anthropic SDK not available; install anthropic")

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not set")

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model or "claude-3-5-haiku-latest",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response text as JSON
        text = response.content[0].text if response.content else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("Claude returned invalid JSON")
            else:
                raise ProviderError("Claude returned invalid JSON")

        cost = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_cost": 0.0,  # Anthropic billing is per-token, calculate offline if needed
            "model": model or "claude-3-5-haiku-latest"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Claude."""
        return self.score(prompt, model)  # Same endpoint for both stages


class OpenAIProvider(BaseProvider):
    """OpenAI API (GPT-4o mini for cost efficiency)."""

    def name(self) -> str:
        return "openai"

    def available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using OpenAI."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK not available; install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("OpenAI returned invalid JSON")
            else:
                raise ProviderError("OpenAI returned invalid JSON")

        cost = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": 0.0,
            "model": model or "gpt-4o-mini"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using OpenAI."""
        return self.score(prompt, model)


class DeepSeekProvider(BaseProvider):
    """DeepSeek API (very cheap option)."""

    def name(self) -> str:
        return "deepseek"

    def available(self) -> bool:
        return bool(os.getenv("DEEPSEEK_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using DeepSeek."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK (used for DeepSeek) not available")

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ProviderError("DEEPSEEK_API_KEY not set")

        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model=model or "deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("DeepSeek returned invalid JSON")
            else:
                raise ProviderError("DeepSeek returned invalid JSON")

        cost = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": 0.0,
            "model": model or "deepseek-chat"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using DeepSeek."""
        return self.score(prompt, model)


class GrokProvider(BaseProvider):
    """Grok (xAI) API."""

    def name(self) -> str:
        return "grok"

    def available(self) -> bool:
        return bool(os.getenv("XAI_API_KEY"))

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Grok."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ProviderError("OpenAI SDK (used for Grok) not available")

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ProviderError("XAI_API_KEY not set")

        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        response = client.chat.completions.create(
            model=model or "grok-2-1212",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        text = response.choices[0].message.content if response.choices else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                else:
                    raise ProviderError("Grok returned invalid JSON")
            else:
                raise ProviderError("Grok returned invalid JSON")

        cost = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_cost": 0.0,
            "model": model or "grok-2-1212"
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Grok."""
        return self.score(prompt, model)


class BedrockNovaProvider(BaseProvider):
    """Amazon Bedrock - Amazon Nova Lite (via AWS credentials)."""

    def name(self) -> str:
        return "bedrock-nova"

    def available(self) -> bool:
        # Bedrock uses AWS credentials from environment or IAM role
        # No explicit API key needed, but boto3 must be available
        try:
            import boto3
            return True
        except ImportError:
            return False

    def score(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Score windows using Amazon Nova on Bedrock."""
        try:
            import boto3
            import json as json_module
        except ImportError:
            raise ProviderError("boto3 not available for Bedrock")

        region = os.getenv("AWS_REGION", "us-east-1")
        client = boto3.client("bedrock-runtime", region_name=region)

        model_id = model or "amazon.nova-lite-v1:0"
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096
        }

        response = client.invoke_model(
            modelId=model_id,
            body=json_module.dumps(payload),
            contentType="application/json"
        )

        response_body = json_module.loads(response["body"].read())
        text = response_body.get("content", [{}])[0].get("text", "{}")

        try:
            parsed = json_module.loads(text)
        except json_module.JSONDecodeError:
            if "```" in text:
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json_module.loads(text[start:end])
                else:
                    raise ProviderError("Bedrock returned invalid JSON")
            else:
                raise ProviderError("Bedrock returned invalid JSON")

        # Bedrock doesn't return token counts in Nova, estimate or leave as 0
        cost = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0,
            "model": model_id
        }
        return parsed, cost

    def detail(self, prompt: str, model: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract detailed clips using Amazon Nova on Bedrock."""
        return self.score(prompt, model)


# Provider registry
PROVIDERS = {
    "gemini": GeminiProvider(),
    "claude": ClaudeProvider(),
    "openai": OpenAIProvider(),
    "deepseek": DeepSeekProvider(),
    "grok": GrokProvider(),
    "bedrock-nova": BedrockNovaProvider(),
}

# Default fallback order: Gemini first (free), then cheaper alternatives
DEFAULT_FALLBACK_ORDER = "gemini,claude,deepseek,openai,grok,bedrock-nova"


def get_fallback_chain() -> List[str]:
    """Parse LLM_FALLBACK_ORDER env var or use default."""
    order = os.getenv("LLM_FALLBACK_ORDER", DEFAULT_FALLBACK_ORDER)
    return [p.strip() for p in order.split(",") if p.strip()]


def call_with_fallback(
    callback,
    model_name: str,
    stage_label: str = "stage"
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], str]:
    """
    Call callback(provider, model_name) across fallback chain until success.
    Returns (parsed_response, cost_analysis, provider_name) or (None, None, "") on failure.

    callback: function(provider: BaseProvider, model: str) -> (parsed, cost)
    model_name: the model to request from each provider
    stage_label: for logging (e.g., "score" or "detail")
    """
    chain = get_fallback_chain()
    errors = []

    for provider_name in chain:
        if provider_name not in PROVIDERS:
            logger.warning(f"Unknown provider in fallback chain: {provider_name}")
            continue

        provider = PROVIDERS[provider_name]
        if not provider.available():
            logger.debug(f"Provider {provider_name} not available (missing API key?), skipping")
            continue

        try:
            logger.info(f"Trying {provider_name} for {stage_label}...")
            parsed, cost = callback(provider, model_name)
            logger.info(f"✅ {provider_name} succeeded for {stage_label}")
            return parsed, cost, provider_name
        except Exception as e:
            error_msg = str(e)[:200]
            logger.warning(f"❌ {provider_name} failed for {stage_label}: {error_msg}")
            errors.append(f"{provider_name}: {error_msg}")

    logger.error(f"All providers exhausted for {stage_label}. Errors: {'; '.join(errors)}")
    return None, None, ""


def get_model_for_provider(provider_name: str) -> str:
    """Get the default model name for a provider."""
    defaults = {
        "gemini": "gemini-3.1-flash-lite",
        "claude": "claude-3-5-haiku-latest",
        "openai": "gpt-4o-mini",
        "deepseek": "deepseek-chat",
        "grok": "grok-2-1212",
        "bedrock-nova": "amazon.nova-lite-v1:0",
    }
    return defaults.get(provider_name, "")
