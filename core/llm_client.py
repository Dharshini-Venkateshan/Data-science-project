"""
LLM Client abstraction for Autonomous AI Data Scientist.
Modular client interface with Google Gemini as the primary provider.
"""

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Optional, Any, Dict
from pydantic import BaseModel, ValidationError

from core.config import GEMINI_API_KEY, DEFAULT_REASONING_MODEL, DEFAULT_FAST_MODEL

T = TypeVar("T", bound=BaseModel)


class BaseLLMClient(ABC):
    """Abstract base interface for LLM providers."""

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        model_name: Optional[str] = None
    ) -> str:
        """Generates plain text response."""
        pass

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        model_name: Optional[str] = None
    ) -> T:
        """Generates structured response validated against a Pydantic schema."""
        pass


def _clean_json_string(text: str) -> str:
    """Strips markdown code blocks and whitespace from a JSON response."""
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


class GeminiLLMClient(BaseLLMClient):
    """
    Google Gemini client implementation using the official google-genai SDK.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            raise ValueError(
                "Gemini API key is missing or not configured. "
                "Please set GEMINI_API_KEY in your .env file or environment, "
                "or pass api_key directly to GeminiLLMClient."
            )

        self.default_model = default_model or DEFAULT_REASONING_MODEL
        
        # Initialize Google GenAI client
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """Generates a text completion using Gemini."""
        from google.genai import types

        model = model_name or self.default_model
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )
                return response.text or ""
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                time.sleep(wait_time)

        raise RuntimeError(f"Gemini generate_text failed after {max_retries} attempts: {last_error}")

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        model_name: Optional[str] = None,
        max_retries: int = 3
    ) -> T:
        """
        Generates structured data conforming to the supplied Pydantic schema.
        Employs native response_schema and secondary validation with auto-repair.
        """
        from google.genai import types

        model = model_name or self.default_model
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=response_schema
        )

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )

                # 1. Check if the SDK provided an already parsed object
                if hasattr(response, "parsed") and isinstance(response.parsed, response_schema):
                    return response.parsed

                # 2. Extract and parse raw text
                raw_text = response.text or ""
                cleaned_text = _clean_json_string(raw_text)

                # 3. Validate against Pydantic model
                return response_schema.model_validate_json(cleaned_text)

            except (ValidationError, json.JSONDecodeError) as parse_err:
                last_error = parse_err
                # Retry with explicit guidance
                prompt = (
                    f"{prompt}\n\n"
                    f"CRITICAL: The previous output failed schema validation with error: {parse_err}. "
                    f"Return ONLY valid JSON matching the exact schema."
                )
                time.sleep(1)

            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Gemini generate_structured failed after {max_retries} attempts: {last_error}")


def get_llm_client(
    provider: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> BaseLLMClient:
    """
    Factory function for obtaining an LLM client.
    Keeps provider selection modular while focusing on Gemini for Phase 1.
    """
    provider = provider.lower().strip()
    if provider == "gemini":
        return GeminiLLMClient(api_key=api_key, default_model=model_name)
    else:
        raise NotImplementedError(
            f"Provider '{provider}' is not supported in Phase 1. "
            f"Only 'gemini' is supported."
        )
