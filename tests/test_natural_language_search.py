"""Tests for natural language search functionality."""

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


class TestNaturalLanguagePromptConstruction:
    """Test NL search prompt building."""

    def test_prompt_contains_query(self, sample_movies):
        """Test that the NL prompt includes the user's query."""
        client = make_client()

        query = "movies about found family"
        system_prompt, user_prompt = client._build_natural_language_prompt(
            sample_movies, query
        )

        assert "found family" in user_prompt
        assert "collection_name" in system_prompt
        assert "matching_indices" in system_prompt

    def test_prompt_contains_movie_data(self, sample_movies):
        """Test that the NL prompt includes movie metadata."""
        client = make_client()

        system_prompt, user_prompt = client._build_natural_language_prompt(
            sample_movies[:3], "any query"
        )

        assert "Paddington" in user_prompt
        assert "Inception" in user_prompt
        assert "John Wick" in user_prompt


class TestNaturalLanguageResponseParsing:
    """Test NL search response parsing."""

    @pytest.mark.asyncio
    async def test_basic_nl_search(self, sample_movies):
        """Test basic NL search with mock response."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            mock_response = Mock()
            mock_response.usage.total_tokens = 500

            mock_choice = Mock()
            mock_choice.message.content = json.dumps({
                "collection_name": "Found Family Films",
                "matching_indices": [1, 4],
                "reasoning": "Movies featuring themes of found family and belonging"
            })
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_client_instance.chat.completions.create.return_value = mock_response

            with patch('api.openai_client.get_config') as mock_config:
                mock_cfg = Mock()
                mock_cfg.ai.api_key = "test"
                mock_cfg.ai.model = "gpt-4o-mini"
                mock_cfg.ai.max_tokens = 4000
                mock_cfg.ai.temperature = 0.3
                mock_cfg.ai.batch_size = 10
                mock_cfg.ai.rate_limit.requests_per_minute = 20
                mock_cfg.ai.rate_limit.tokens_per_minute = 40000
                mock_config.return_value = mock_cfg

                client = OpenAIClient(api_key="test", model="gpt-4o-mini")

            result = await client.natural_language_search(sample_movies, "movies about found family")

            assert result.success
            assert result.suggested_name == "Found Family Films"
            assert len(result.results) == 2

            matched_ids = {r.media_item_id for r in result.results}
            assert 1 in matched_ids  # Paddington
            assert 4 in matched_ids  # The Holiday

    @pytest.mark.asyncio
    async def test_empty_nl_search_results(self, sample_movies):
        """Test NL search that returns no matches."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            mock_response = Mock()
            mock_response.usage.total_tokens = 300

            mock_choice = Mock()
            mock_choice.message.content = json.dumps({
                "collection_name": "Underwater Horror",
                "matching_indices": [],
                "reasoning": "No movies in library match underwater horror themes"
            })
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_client_instance.chat.completions.create.return_value = mock_response

            with patch('api.openai_client.get_config') as mock_config:
                mock_cfg = Mock()
                mock_cfg.ai.api_key = "test"
                mock_cfg.ai.model = "gpt-4o-mini"
                mock_cfg.ai.max_tokens = 4000
                mock_cfg.ai.temperature = 0.3
                mock_cfg.ai.batch_size = 10
                mock_cfg.ai.rate_limit.requests_per_minute = 20
                mock_cfg.ai.rate_limit.tokens_per_minute = 40000
                mock_config.return_value = mock_cfg

                client = OpenAIClient(api_key="test", model="gpt-4o-mini")

            result = await client.natural_language_search(sample_movies, "underwater horror movies")

            assert result.success
            assert len(result.results) == 0
            assert result.suggested_name == "Underwater Horror"

    @pytest.mark.asyncio
    async def test_nl_search_invalid_indices_handled(self, sample_movies):
        """Test that invalid indices in NL response are gracefully handled."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            mock_response = Mock()
            mock_response.usage.total_tokens = 400

            mock_choice = Mock()
            mock_choice.message.content = json.dumps({
                "collection_name": "Test",
                "matching_indices": [1, 999, -1, "not_a_number", 3],
                "reasoning": "Test"
            })
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_client_instance.chat.completions.create.return_value = mock_response

            with patch('api.openai_client.get_config') as mock_config:
                mock_cfg = Mock()
                mock_cfg.ai.api_key = "test"
                mock_cfg.ai.model = "gpt-4o-mini"
                mock_cfg.ai.max_tokens = 4000
                mock_cfg.ai.temperature = 0.3
                mock_cfg.ai.batch_size = 10
                mock_cfg.ai.rate_limit.requests_per_minute = 20
                mock_cfg.ai.rate_limit.tokens_per_minute = 40000
                mock_config.return_value = mock_cfg

                client = OpenAIClient(api_key="test", model="gpt-4o-mini")

            result = await client.natural_language_search(sample_movies, "test query")

            assert result.success
            # Only indices 1 and 3 should be valid
            assert len(result.results) == 2
            matched_ids = {r.media_item_id for r in result.results}
            assert 1 in matched_ids
            assert 3 in matched_ids


class TestNaturalLanguageChunking:
    """Test chunked NL search for large libraries."""

    @pytest.mark.asyncio
    async def test_chunked_search_triggers_for_large_libraries(self):
        """Test that chunking is used for libraries exceeding the limit."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            large_library = []
            for i in range(50):
                large_library.append({
                    "id": i + 1,
                    "title": f"Movie {i+1}",
                    "year": 2000 + i,
                    "type": "movie",
                    "summary": f"Summary for movie {i+1}",
                    "genres": ["Drama"],
                    "directors": [f"Director {i+1}"]
                })

            mock_response = Mock()
            mock_response.usage.total_tokens = 500
            mock_choice = Mock()
            mock_choice.message.content = json.dumps({
                "collection_name": "Test Collection",
                "matching_indices": [1, 2],
                "reasoning": "Test"
            })
            mock_choice.finish_reason = "stop"
            mock_response.choices = [mock_choice]
            mock_client_instance.chat.completions.create.return_value = mock_response

            with patch('api.openai_client.get_config') as mock_config:
                mock_cfg = Mock()
                mock_cfg.ai.api_key = "test"
                mock_cfg.ai.model = "gpt-4o-mini"
                mock_cfg.ai.max_tokens = 4000
                mock_cfg.ai.temperature = 0.3
                mock_cfg.ai.batch_size = 10
                mock_cfg.ai.rate_limit.requests_per_minute = 20
                mock_cfg.ai.rate_limit.tokens_per_minute = 40000
                mock_config.return_value = mock_cfg

                client = OpenAIClient(api_key="test", model="gpt-4o-mini")

            result = await client.natural_language_search(large_library, "test query")

            assert result.success
            # Should have made multiple API calls (chunked)
            assert mock_client_instance.chat.completions.create.call_count >= 2


class TestNaturalLanguageCollectionDB:
    """Test NL collection database operations."""

    def test_create_nl_collection(self, test_session, test_db_manager):
        """Test creating a natural language collection in the database."""
        category = test_db_manager.create_natural_language_collection(
            test_session,
            name="Rainy Day Movies",
            query="cozy movies perfect for a rainy day"
        )

        assert category.name == "Rainy Day Movies"
        assert category.source == "natural_language"
        assert category.natural_language_query == "cozy movies perfect for a rainy day"
        assert "cozy movies" in category.description

    def test_get_nl_collections(self, test_session, test_db_manager):
        """Test retrieving natural language collections."""
        test_db_manager.create_collection_category(
            test_session, "Regular Category", "Regular desc", "Regular prompt",
            source="config"
        )
        test_db_manager.create_natural_language_collection(
            test_session, "NL Category 1", "query 1"
        )
        test_db_manager.create_natural_language_collection(
            test_session, "NL Category 2", "query 2"
        )

        nl_collections = test_db_manager.get_natural_language_collections(test_session)

        assert len(nl_collections) == 2
        names = {c.name for c in nl_collections}
        assert "NL Category 1" in names
        assert "NL Category 2" in names
        assert "Regular Category" not in names

    def test_nl_collection_in_stats(self, test_session, test_db_manager):
        """Test that NL collections show up in database stats."""
        test_db_manager.create_natural_language_collection(
            test_session, "Test NL", "test query"
        )

        stats = test_db_manager.get_database_stats(test_session)
        assert stats.get("nl_collections", 0) == 1
