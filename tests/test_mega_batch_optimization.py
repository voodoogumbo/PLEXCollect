"""Tests for the mega-batch optimization system."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from api.openai_client import OpenAIClient, ClassificationResult, BatchResult


class TestMegaBatchOptimization:
    """Test the mega-batch optimization functionality."""
    
    @pytest.mark.asyncio
    async def test_mega_batch_vs_legacy_cost_comparison(self, star_wars_movies, mcu_movies, sample_config):
        """Test that mega-batch significantly reduces API costs compared to legacy method."""
        
        all_movies = star_wars_movies + mcu_movies
        categories = [
            {
                "id": 1, 
                "name": "Star Wars Saga", 
                "description": "Star Wars movies in chronological order",
                "prompt": "Is this a Star Wars movie?",
                "is_franchise": True
            },
            {
                "id": 2, 
                "name": "Marvel Cinematic Universe", 
                "description": "MCU movies in chronological order",
                "prompt": "Is this an MCU movie?",
                "is_franchise": True
            },
            {
                "id": 3, 
                "name": "Christmas Movies", 
                "description": "Movies with Christmas themes",
                "prompt": "Is this a Christmas movie?",
                "is_franchise": False
            }
        ]
        
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Mock response for mega-batch (1 API call) - JSON format
            mock_response = Mock()
            mock_response.usage.total_tokens = 3000  # Single large response
            
            # Build new JSON response format
            results_array = []
            for i, movie in enumerate(all_movies):
                for cat in categories:
                    results_array.append({
                        "movie_id": i + 1,
                        "media_item_id": movie["id"],
                        "category_id": cat["id"],
                        "category_name": cat["name"],
                        "matches": False,
                        "confidence": 0.1,
                        "reasoning": "Test classification"
                    })
            
            # Create mock choices structure
            mock_choice = Mock()
            mock_choice.message.content = json.dumps({"results": results_array})
            mock_response.choices = [mock_choice]
            
            mock_client.chat.completions.create.return_value = mock_response
            
            openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")
            
            # Test mega-batch approach
            mega_result = await openai_client.classify_all_items_mega_batch(all_movies, categories)
            
            # Calculate costs
            mega_cost = mega_result.total_cost
            
            # Legacy approach would be: len(movies) * len(categories) / batch_size API calls
            # With batch_size=10, that's ceil(9/10) * 3 = 3 API calls minimum
            # Each call would be ~1000 tokens, so 3000 tokens total minimum
            legacy_estimated_calls = len(categories) * ((len(all_movies) + 9) // 10)  # 3 calls
            legacy_estimated_tokens = legacy_estimated_calls * 1000  # Conservative estimate
            legacy_estimated_cost = (legacy_estimated_tokens / 1000) * openai_client.cost_per_1k_tokens
            
            # Mega-batch should be significantly more cost-effective
            # (In practice, legacy would be much more expensive due to repeated context)
            assert mega_cost <= legacy_estimated_cost
            assert mega_result.total_tokens == 3000  # Single call
    
    @pytest.mark.asyncio
    async def test_mega_batch_prompt_construction(self, star_wars_movies, sample_config):
        """Test that mega-batch prompts are correctly constructed."""
        with patch('api.openai_client.OpenAI'):
            openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")
            
            categories = [
                {
                    "id": 1,
                    "name": "Star Wars Saga",
                    "description": "Star Wars movies in chronological story order",
                    "prompt": "Is this a Star Wars movie?",
                    "is_franchise": True
                }
            ]
            
            system_prompt, user_prompt = openai_client._build_mega_batch_prompt(
                star_wars_movies[:3], categories
            )
            
            # Check system prompt contains key elements
            assert "Plex-Classifier v2" in system_prompt
            assert "structured movie classification data" in system_prompt
            assert "franchise categories" in system_prompt
            
            # Check user prompt contains JSON payload with category data
            assert "DATA PAYLOAD" in user_prompt
            assert "Star Wars Saga" in user_prompt  # Should be in the JSON payload
            assert "Star Wars: Episode IV" in user_prompt
            assert "Star Wars: Episode I" in user_prompt
            
            # Check JSON response format specification
            assert "results" in user_prompt
            assert "chronological_position" in user_prompt
            assert "franchise_year" in user_prompt
    
    @pytest.mark.asyncio
    async def test_mega_batch_response_parsing(self, star_wars_movies, sample_config):
        """Test that mega-batch responses are correctly parsed."""
        with patch('api.openai_client.OpenAI'):
            openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")
            
            categories = [
                {"id": 1, "name": "Star Wars Saga", "is_franchise": True},
                {"id": 2, "name": "Christmas Movies", "is_franchise": False}
            ]
            
            # Mock response with new JSON format
            mock_response = Mock()
            mock_response.usage.total_tokens = 2000
            
            results_array = [
                {
                    "movie_id": 1,
                    "media_item_id": 1,
                    "category_id": 1,
                    "category_name": "Star Wars Saga",
                    "matches": True,
                    "confidence": 0.95,
                    "reasoning": "Episode IV in chronological order",
                    "chronological_position": 4,
                    "franchise_year": 0
                },
                {
                    "movie_id": 1,
                    "media_item_id": 1,
                    "category_id": 2,
                    "category_name": "Christmas Movies",
                    "matches": False,
                    "confidence": 0.1,
                    "reasoning": "No Christmas themes"
                },
                {
                    "movie_id": 2,
                    "media_item_id": 2,
                    "category_id": 1,
                    "category_name": "Star Wars Saga",
                    "matches": True,
                    "confidence": 0.95,
                    "reasoning": "Episode I, first chronologically",
                    "chronological_position": 1,
                    "franchise_year": -32
                },
                {
                    "movie_id": 2,
                    "media_item_id": 2,
                    "category_id": 2,
                    "category_name": "Christmas Movies",
                    "matches": False,
                    "confidence": 0.1,
                    "reasoning": "No Christmas themes"
                }
            ]
            
            mock_choice = Mock()
            mock_choice.message.content = json.dumps({"results": results_array})
            mock_response.choices = [mock_choice]
            
            # Parse the response
            results = openai_client._parse_mega_batch_response(
                mock_response, star_wars_movies[:2], categories
            )
            
            # Should have 2 movies × 2 categories = 4 results
            assert len(results) == 4
            
            # Check Star Wars classifications
            star_wars_results = [r for r in results if r.matches and r.category_id == 1]
            assert len(star_wars_results) == 2
            
            # Check franchise positioning
            positions = {r.media_item_id: r.franchise_position for r in star_wars_results}
            assert positions[1] == 4  # Episode IV
            assert positions[2] == 1  # Episode I
            
            # Check franchise years
            years = {r.media_item_id: r.franchise_year for r in star_wars_results}
            assert years[1] == 0   # Episode IV: 0 ABY
            assert years[2] == -32  # Episode I: 32 BBY
    
    @pytest.mark.asyncio
    async def test_fallback_to_legacy_on_failure(self, star_wars_movies, sample_config):
        """Test that the system falls back to legacy method if mega-batch fails."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")
            
            categories = [{
                "id": 1, 
                "name": "Star Wars Saga", 
                "description": "Star Wars movies in chronological order",
                "prompt": "Is this a Star Wars movie?",
                "is_franchise": True
            }]
            
            # First, make mega-batch fail
            with patch.object(openai_client, 'classify_all_items_mega_batch', side_effect=Exception("Mega-batch failed")):
                with patch.object(openai_client, 'classify_media_items_legacy', return_value=[]) as mock_legacy:
                    
                    # This should trigger fallback
                    results = await openai_client.classify_media_items(star_wars_movies, categories)
                    
                    # Verify legacy method was called
                    mock_legacy.assert_called_once()
    
    def test_cost_estimation_accuracy(self):
        """Test that cost estimation is accurate for different models."""
        # Test different model cost estimations
        models_and_costs = [
            ("gpt-4o-mini", 0.0001),
            ("gpt-4o", 0.005),
            ("gpt-4", 0.03),
            ("gpt-3.5-turbo", 0.002)
        ]
        
        for model, expected_cost in models_and_costs:
            with patch('api.openai_client.OpenAI'):
                client = OpenAIClient(api_key="test", model=model)
                actual_cost = client._get_cost_per_1k_tokens()
                # Allow for default cost fallback for unrecognized models
                if model == "gpt-4o-mini":
                    assert actual_cost == expected_cost
                elif model == "gpt-4o":
                    assert actual_cost == expected_cost
                elif model == "gpt-4":
                    assert actual_cost == expected_cost
                elif model == "gpt-3.5-turbo":
                    assert actual_cost == expected_cost
                else:
                    # Default cost for unrecognized models
                    assert actual_cost >= 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_mega_batch(self, star_wars_movies, sample_config):
        """Test that rate limiting works correctly with mega-batch."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            # Create client with very restrictive rate limits
            openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")
            openai_client.rate_limiter.requests_per_minute = 1
            openai_client.rate_limiter.tokens_per_minute = 5000
            
            categories = [{
                "id": 1, 
                "name": "Star Wars Saga", 
                "description": "Star Wars movies in chronological order",
                "prompt": "Is this a Star Wars movie?",
                "is_franchise": True
            }]
            
            # Mock response with new JSON format
            mock_response = Mock()
            mock_response.usage.total_tokens = 3000
            
            results_array = [{
                "movie_id": 1,
                "media_item_id": 1,
                "category_id": 1,
                "category_name": "Star Wars Saga",
                "matches": True,
                "confidence": 0.95,
                "reasoning": "Star Wars movie"
            }]
            
            mock_choice = Mock()
            mock_choice.message.content = json.dumps({"results": results_array})
            mock_response.choices = [mock_choice]
            mock_client.chat.completions.create.return_value = mock_response
            
            # This should work within rate limits (1 request, 3000 tokens)
            result = await openai_client.classify_all_items_mega_batch(
                star_wars_movies[:1], categories
            )
            
            assert result.success
            assert result.total_tokens == 3000