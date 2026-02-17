"""Tests for collection manager integration."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from api.collection_manager import CollectionManager


class TestCollectionManagerIntegration:
    """Test collection manager integration with vibe-based classification."""

    @pytest.mark.asyncio
    async def test_mega_batch_integration(self, test_db_manager, sample_config):
        """Test that collection manager uses mega-batch optimization."""

        # Create the category in the test database first
        with test_db_manager.get_session() as session:
            test_db_manager.create_collection_category(
                session, "Cozy Comfort Movies",
                "Warm, comforting movies",
                "Does this movie have a cozy vibe?",
                source="config"
            )

        with patch('api.collection_manager.get_plex_client') as mock_plex:
            with patch('api.collection_manager.get_openai_client') as mock_openai:
                with patch('api.collection_manager.get_database_manager', return_value=test_db_manager):

                    # Setup mocks
                    mock_plex_instance = Mock()
                    mock_plex_instance.connect.return_value = True
                    mock_plex_instance.get_library_sections.return_value = [
                        Mock(title="Movies", type="movie")
                    ]
                    mock_plex_instance.scan_library_section.return_value = [
                        {
                            "plex_key": "movie1",
                            "title": "Paddington",
                            "year": 2014,
                            "type": "movie",
                            "library_section": "Movies"
                        }
                    ]
                    mock_plex.return_value = mock_plex_instance

                    mock_openai_instance = Mock()
                    mock_openai_instance.model = "gpt-4o-mini"

                    mega_batch_result = Mock()
                    mega_batch_result.success = True
                    mega_batch_result.total_tokens = 1000
                    mega_batch_result.total_cost = 0.001
                    mega_batch_result.processing_time = 2.0
                    mega_batch_result.error_message = None
                    mega_batch_result.results = [
                        Mock(
                            media_item_id=1,
                            category_id=1,
                            matches=True,
                            confidence=0.85,
                            reasoning="Cozy, feel-good movie",
                            tokens_used=100,
                            processing_time=0.2,
                            error=None
                        )
                    ]

                    mock_openai_instance.classify_media_items = AsyncMock(return_value=[mega_batch_result])
                    mock_openai.return_value = mock_openai_instance

                    with patch('api.collection_manager.get_config', return_value=sample_config):
                        collection_manager = CollectionManager()

                        result = await collection_manager.full_scan_and_classify(
                            library_sections=["Movies"],
                            categories=["Cozy Comfort Movies"]
                        )

                    assert result["ai_requests_made"] == 1
                    assert result["total_cost_estimate"] == 0.001
                    assert result["status"] == "completed"

    def test_vibe_collection_creation(self, test_session, test_db_manager):
        """Test that vibe-based collections are created and items are classified."""

        # Create some movies
        movies_data = [
            ("pad1", "Paddington", 2014),
            ("hol1", "The Holiday", 2006),
            ("jw1", "John Wick", 2014),
        ]

        for plex_key, title, year in movies_data:
            test_db_manager.get_or_create_media_item(
                test_session,
                plex_key=plex_key,
                title=title,
                year=year,
                type="movie",
                library_section="Movies"
            )

        # Create vibe category
        from models.database_models import CollectionCategory
        category = CollectionCategory(
            name="Cozy Comfort Movies",
            description="Warm, comforting movies",
            prompt="Does this movie have a cozy vibe?",
            source="config"
        )
        test_session.add(category)
        test_session.flush()

        # Create classifications
        from models.database_models import ItemClassification, MediaItem
        movies = test_session.query(MediaItem).all()

        # Only Paddington and The Holiday should match "cozy"
        for movie in movies:
            matches = movie.title in ["Paddington", "The Holiday"]
            classification = ItemClassification(
                media_item_id=movie.id,
                category_id=category.id,
                matches=matches,
                confidence=0.85 if matches else 0.1
            )
            test_session.add(classification)

        test_session.commit()

        # Verify correct items matched
        matching_items = test_db_manager.get_category_items(
            test_session, category.id, matches_only=True
        )

        assert len(matching_items) == 2
        titles = {item.title for item in matching_items}
        assert "Paddington" in titles
        assert "The Holiday" in titles
        assert "John Wick" not in titles

    def test_natural_language_collection_creation(self, test_session, test_db_manager):
        """Test creating a collection from natural language query."""

        # Create a NL collection
        nl_category = test_db_manager.create_natural_language_collection(
            test_session,
            name="Found Family Films",
            query="movies about found family and belonging"
        )

        assert nl_category.source == "natural_language"
        assert nl_category.natural_language_query == "movies about found family and belonging"
        assert nl_category.name == "Found Family Films"

        # Verify it shows up in NL collections list
        nl_collections = test_db_manager.get_natural_language_collections(test_session)
        assert len(nl_collections) == 1
        assert nl_collections[0].name == "Found Family Films"


class TestErrorHandling:
    """Test error handling."""

    def test_missing_data_handling(self, test_session, test_db_manager):
        """Test handling of movies with incomplete data."""

        movie, _ = test_db_manager.get_or_create_media_item(
            test_session,
            plex_key="incomplete_movie",
            title="Incomplete Movie",
            year=2000,
            type="movie",
            library_section="Movies"
        )

        # Database queries should work with minimal data
        all_items = test_db_manager.get_media_items(test_session)
        assert len(all_items) == 1

    def test_malformed_api_response_handling(self, sample_config):
        """Test handling of malformed OpenAI API responses."""
        from api.openai_client import OpenAIClient

        with patch('api.openai_client.OpenAI'):
            with patch('api.openai_client.get_config', return_value=sample_config):
                client = OpenAIClient(api_key="test", model="test")

                mock_response = Mock()
                mock_choice = Mock()
                mock_choice.message.content = "Invalid JSON response"
                mock_response.choices = [mock_choice]
                mock_response.usage.total_tokens = 100

                results = client._parse_mega_batch_response(
                    mock_response,
                    [{"id": 1, "title": "Test Movie"}],
                    [{"id": 1, "name": "Test Category"}]
                )

                assert len(results) == 1
                assert results[0].error is not None
                assert "Parse error" in results[0].error

    def test_database_column_migration_safety(self, temp_db_path):
        """Test that missing database columns are handled gracefully."""
        from api.database import DatabaseManager

        db_manager = DatabaseManager(temp_db_path)

        with db_manager.get_session() as session:
            stats = db_manager.get_database_stats(session)
            assert isinstance(stats, dict)

            nl_collections = db_manager.get_natural_language_collections(session)
            assert isinstance(nl_collections, list)
