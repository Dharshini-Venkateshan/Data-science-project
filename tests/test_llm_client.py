"""
Unit tests for the modular LLM Client interface with Google Gemini.
Tests client factory, mock structured output validation, and conditional live calls.
"""

import os
import json
from unittest.mock import MagicMock, patch
import pytest
from pydantic import BaseModel

from core.llm_client import (
    BaseLLMClient,
    GeminiLLMClient,
    get_llm_client,
    _clean_json_string
)
from core.config import GEMINI_API_KEY


class SamplePlanSchema(BaseModel):
    """Test schema for structured output verification."""
    target_column: str
    problem_type: str
    confidence_score: float
    recommended_models: list[str]


def test_clean_json_string():
    """Verifies that markdown fences are stripped cleanly."""
    raw_markdown = '```json\n{"name": "test", "val": 123}\n```'
    assert _clean_json_string(raw_markdown) == '{"name": "test", "val": 123}'

    raw_plain = '{"name": "test", "val": 123}'
    assert _clean_json_string(raw_plain) == '{"name": "test", "val": 123}'


def test_factory_missing_api_key():
    """Verifies that initializing without an API key raises ValueError."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": ""}, clear=True):
        with pytest.raises(ValueError) as excinfo:
            get_llm_client(provider="gemini", api_key="")
        assert "Gemini API key is missing" in str(excinfo.value)


def test_factory_unsupported_provider():
    """Verifies that requesting an unconfigured provider raises NotImplementedError."""
    with pytest.raises(NotImplementedError) as excinfo:
        get_llm_client(provider="openai", api_key="dummy_key")
    assert "Provider 'openai' is not supported in Phase 1" in str(excinfo.value)


def test_gemini_client_instantiation():
    """Verifies successful client instantiation with an explicit key."""
    with patch("google.genai.Client") as mock_genai_client:
        client = get_llm_client(provider="gemini", api_key="test_api_key_123")
        assert isinstance(client, BaseLLMClient)
        assert isinstance(client, GeminiLLMClient)
        mock_genai_client.assert_called_once_with(api_key="test_api_key_123")


def test_gemini_mocked_generate_text():
    """Verifies text generation using mocked GenAI client."""
    with patch("google.genai.Client") as mock_genai_client:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Data science plan approved."
        mock_instance.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_instance

        client = GeminiLLMClient(api_key="fake_key_abc")
        response = client.generate_text("Analyze this dataset.")

        assert response == "Data science plan approved."
        mock_instance.models.generate_content.assert_called_once()


def test_gemini_mocked_generate_structured():
    """Verifies structured output parsing into Pydantic model with mocked client."""
    expected_data = {
        "target_column": "churn",
        "problem_type": "binary_classification",
        "confidence_score": 0.95,
        "recommended_models": ["LightGBM", "XGBoost", "RandomForest"]
    }

    with patch("google.genai.Client") as mock_genai_client:
        mock_instance = MagicMock()
        mock_response = MagicMock()
        # Test markdown code block handling as well
        mock_response.text = f"```json\n{json.dumps(expected_data)}\n```"
        mock_instance.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_instance

        client = GeminiLLMClient(api_key="fake_key_abc")
        result = client.generate_structured(
            prompt="Analyze problem type",
            response_schema=SamplePlanSchema
        )

        assert isinstance(result, SamplePlanSchema)
        assert result.target_column == "churn"
        assert result.problem_type == "binary_classification"
        assert result.confidence_score == 0.95
        assert len(result.recommended_models) == 3


@pytest.mark.skipif(
    not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here",
    reason="Requires live GEMINI_API_KEY configured in environment or .env"
)
def test_live_gemini_api_call():
    """
    Live integration test executed only when a real GEMINI_API_KEY is supplied.
    Verifies actual Gemini response generation and structured parsing.
    """
    client = get_llm_client(provider="gemini")
    structured_result = client.generate_structured(
        prompt="Identify the classification type for a dataset with target column 'Survived' with values 0 or 1.",
        response_schema=SamplePlanSchema
    )
    assert isinstance(structured_result, SamplePlanSchema)
    assert structured_result.target_column is not None
    assert len(structured_result.recommended_models) > 0
