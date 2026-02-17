"""Tests for the mega-batch optimization system."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from api.openai_client import OpenAIClient, ClassificationResult, BatchResult


def make_client(model="gpt-4o-mini", sample_config=None):
    """Helper to create an OpenAIClient with config mocked."""
    with patch('api.openai_client.OpenAI'):
        with patch('api.openai_client.get_config') as mock_config:
            if sample_config:
                mock_config.return_value = sample_config
            else:
                # Create a minimal mock config
                mock_cfg = Mock()
                mock_cfg.ai.api_key = "test_key"
                mock_cfg.ai.model = model
                mock_cfg.ai.max_tokens = 4000
                mock_cfg.ai.temperature = 0.3
                mock_cfg.ai.batch_size = 10
                mock_cfg.ai.rate_limit.requests_per_minute = 20
                mock_cfg.ai.rate_limit.tokens_per_minute = 40000
                mock_config.return_value = mock_cfg
            return OpenAIClient(api_key="test_key", model=model)


class TestMegaBatchOptimization:
    """Test the mega-batch optimization functionality."""

    @pytest.mark.asyncio
    async def test_mega_batch_vs_legacy_cost_comparison(self, sample_movies, sample_config):
        """Test that mega-batch significantly reduces API costs compared to legacy method."""

        categories = [
            {"id": 1, "name": "Cozy Comfort Movies", "description": "Warm, comforting movies", "prompt": "Does this movie have a cozy vibe?"},
            {"id": 2, "name": "Mind-Bending Movies", "description": "Movies that question reality", "prompt": "Is this a mind-bending movie?"},
            {"id": 3, "name": "Adrenaline Rush", "description": "Non-stop action", "prompt": "Is this a high-adrenaline movie?"}
        ]

        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_response = Mock()
            mock_response.usage.total_tokens = 3000

            mock_choice = Mock()
            mock_choice.message.content = json.dumps({
                "Cozy Comfort Movies": [1, 4],
                "Mind-Bending Movies": [2, 5],
                "Adrenaline Rush": [3, 6]
            })
            mock_response.choices = [mock_choice]

            mock_client.chat.completions.create.return_value = mock_response

            with patch('api.openai_client.get_config', return_value=sample_config):
                openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")

            mega_result = await openai_client.classify_all_items_mega_batch(sample_movies, categories)

            mega_cost = mega_result.total_cost

            legacy_estimated_calls = len(categories) * ((len(sample_movies) + 9) // 10)
            legacy_estimated_tokens = legacy_estimated_calls * 1000
            legacy_estimated_cost = (legacy_estimated_tokens / 1000) * openai_client.cost_per_1k_tokens

            assert mega_cost <= legacy_estimated_cost
            assert mega_result.total_tokens == 3000

    @pytest.mark.asyncio
    async def test_mega_batch_prompt_construction(self, sample_movies, sample_config):
        """Test that mega-batch prompts are correctly constructed."""

        openai_client = make_client(sample_config=sample_config)

        categories = [
            {"id": 1, "name": "Cozy Comfort Movies", "description": "Warm, comforting movies", "prompt": "Does this movie have a cozy vibe?"}
        ]

        system_prompt, user_prompt = openai_client._build_mega_batch_prompt(
            sample_movies[:3], categories
        )

        assert "Plex-Classifier v5" in system_prompt
        assert "subjective" in system_prompt
        assert "vibe" in system_prompt

        assert "DATA PAYLOAD" in user_prompt
        assert "Cozy Comfort Movies" in user_prompt
        assert "Paddington" in user_prompt
        assert "Inception" in user_prompt

    @pytest.mark.asyncio
    async def test_mega_batch_response_parsing(self, sample_movies, sample_config):
        """Test that mega-batch responses are correctly parsed."""

        openai_client = make_client(sample_config=sample_config)

        categories = [
            {"id": 1, "name": "Cozy Comfort Movies"},
            {"id": 2, "name": "Adrenaline Rush"}
        ]

        mock_response = Mock()
        mock_response.usage.total_tokens = 2000

        mock_choice = Mock()
        mock_choice.message.content = json.dumps({
            "Cozy Comfort Movies": [1, 4],
            "Adrenaline Rush": [3, 6]
        })
        mock_response.choices = [mock_choice]

        results = openai_client._parse_mega_batch_response(
            mock_response, sample_movies, categories
        )

        assert len(results) == len(sample_movies) * len(categories)

        cozy_matches = [r for r in results if r.matches and r.category_id == 1]
        assert len(cozy_matches) == 2
        cozy_ids = {r.media_item_id for r in cozy_matches}
        assert 1 in cozy_ids
        assert 4 in cozy_ids

        action_matches = [r for r in results if r.matches and r.category_id == 2]
        assert len(action_matches) == 2
        action_ids = {r.media_item_id for r in action_matches}
        assert 3 in action_ids
        assert 6 in action_ids

    @pytest.mark.asyncio
    async def test_fallback_to_legacy_on_failure(self, sample_movies, sample_config):
        """Test that the system falls back to legacy method if mega-batch fails."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            with patch('api.openai_client.get_config', return_value=sample_config):
                openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")

            categories = [{"id": 1, "name": "Cozy Comfort Movies", "description": "Warm, comforting movies", "prompt": "Does this movie have a cozy vibe?"}]

            with patch.object(openai_client, 'classify_all_items_mega_batch', side_effect=Exception("Mega-batch failed")):
                with patch.object(openai_client, 'classify_media_items_legacy', return_value=[]) as mock_legacy:
                    results = await openai_client.classify_media_items(sample_movies, categories)
                    mock_legacy.assert_called_once()

    def test_cost_estimation_accuracy(self, sample_config):
        """Test that cost estimation is accurate for different models."""
        models_and_costs = [
            ("gpt-4o-mini", 0.00015),
            ("gpt-4o", 0.005),
            ("gpt-4", 0.03),
            ("gpt-3.5-turbo", 0.002)
        ]

        for model, expected_cost in models_and_costs:
            client = make_client(model=model, sample_config=sample_config)
            actual_cost = client._get_cost_per_1k_tokens()
            assert actual_cost == expected_cost, f"Model {model}: expected {expected_cost}, got {actual_cost}"

    @pytest.mark.asyncio
    async def test_rate_limiting_with_mega_batch(self, sample_movies, sample_config):
        """Test that rate limiting works correctly with mega-batch."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client

            with patch('api.openai_client.get_config', return_value=sample_config):
                openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")

            openai_client.rate_limiter.requests_per_minute = 1
            openai_client.rate_limiter.tokens_per_minute = 5000

            categories = [{"id": 1, "name": "Cozy Comfort Movies", "description": "Warm, comforting movies", "prompt": "Does this movie have a cozy vibe?"}]

            mock_response = Mock()
            mock_response.usage.total_tokens = 3000

            mock_choice = Mock()
            mock_choice.message.content = json.dumps({"Cozy Comfort Movies": [1]})
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response

            result = await openai_client.classify_all_items_mega_batch(
                sample_movies[:1], categories
            )

            assert result.success
            assert result.total_tokens == 3000
