"""Tests for collection manager integration with franchise system."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from api.collection_manager import CollectionManager


class TestCollectionManagerIntegration:
    """Test collection manager integration with franchise features."""
    
    @pytest.mark.asyncio
    async def test_mega_batch_integration(self, test_db_manager, sample_config):
        """Test that collection manager uses mega-batch optimization."""
        
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
                            "title": "Star Wars: Episode IV",
                            "year": 1977,
                            "type": "movie"
                        }
                    ]
                    mock_plex.return_value = mock_plex_instance
                    
                    # Setup OpenAI mock
                    mock_openai_instance = Mock()
                    
                    # Mock mega-batch result
                    mega_batch_result = Mock()
                    mega_batch_result.success = True
                    mega_batch_result.total_tokens = 1000
                    mega_batch_result.total_cost = 0.1
                    mega_batch_result.processing_time = 2.0
                    mega_batch_result.results = [
                        Mock(
                            media_item_id=1,
                            category_id=1,
                            matches=True,
                            confidence=0.95,
                            reasoning="Star Wars movie",
                            tokens_used=100,
                            processing_time=0.2,
                            franchise_position=4,
                            franchise_year=0,
                            franchise_reasoning="Episode IV chronologically",
                            error=None
                        )
                    ]
                    
                    mock_openai_instance.classify_media_items.return_value = [mega_batch_result]
                    mock_openai.return_value = mock_openai_instance
                    
                    # Create collection manager
                    collection_manager = CollectionManager()
                    
                    # Run full scan
                    result = await collection_manager.full_scan_and_classify(
                        library_sections=["Movies"],
                        categories=["Star Wars Saga"]
                    )
                    
                    # Verify mega-batch was used (single API call)
                    assert result["ai_requests_made"] == 1
                    assert result["total_cost_estimate"] == 0.1
                    assert result["status"] == "completed"
    
    def test_franchise_collection_ordering(self, test_session, test_db_manager):
        """Test that franchise collections are ordered chronologically."""
        
        # Create franchise movies in database
        movies_data = [
            ("sw4", "Star Wars: Episode IV", 1977, 4, 0),
            ("sw1", "Star Wars: Episode I", 1999, 1, -32),
            ("sw5", "Star Wars: Episode V", 1980, 5, 3),
            ("sw2", "Star Wars: Episode II", 2002, 2, -22),
        ]
        
        movie_objects = []
        for plex_key, title, year, chrono_order, franchise_year in movies_data:
            movie, _ = test_db_manager.get_or_create_media_item(
                test_session,
                plex_key=plex_key,
                title=title,
                year=year,
                type="movie",
                franchise_name="Star Wars Saga",
                chronological_order=chrono_order,
                franchise_year=franchise_year
            )
            movie_objects.append(movie)
        
        # Create franchise category
        from models.database_models import CollectionCategory
        category = CollectionCategory(
            name="Star Wars Saga",
            description="Star Wars movies in chronological order",
            prompt="Is this a Star Wars movie?",
            is_franchise=True,
            chronological_sorting=True
        )
        test_session.add(category)
        test_session.flush()
        
        # Create classifications
        from models.database_models import ItemClassification
        for movie in movie_objects:
            classification = ItemClassification(
                media_item_id=movie.id,
                category_id=category.id,
                matches=True,
                confidence=0.95
            )
            test_session.add(classification)
        
        test_session.commit()
        
        # Test chronological ordering
        ordered_movies = test_db_manager.get_franchise_movies(
            test_session, "Star Wars Saga", ordered=True
        )
        
        # Should be ordered: Episode I, II, IV, V
        expected_order = [1, 2, 4, 5]
        actual_order = [movie.chronological_order for movie in ordered_movies]
        assert actual_order == expected_order
        
        # Titles should also be in correct order
        titles = [movie.title for movie in ordered_movies]
        assert "Episode I" in titles[0]
        assert "Episode II" in titles[1]
        assert "Episode IV" in titles[2]
        assert "Episode V" in titles[3]
    
    @pytest.mark.asyncio
    async def test_franchise_info_storage(self, test_db_manager, sample_config):
        """Test that franchise information is properly stored during classification."""
        
        with patch('api.collection_manager.get_plex_client'):
            with patch('api.collection_manager.get_openai_client') as mock_openai:
                with patch('api.collection_manager.get_database_manager', return_value=test_db_manager):
                    
                    # Create a movie in the database first
                    with test_db_manager.get_session() as session:
                        movie, _ = test_db_manager.get_or_create_media_item(
                            session,
                            plex_key="test_movie",
                            title="Star Wars: Episode IV",
                            year=1977,
                            type="movie"
                        )
                        movie_id = movie.id
                    
                    # Setup OpenAI mock with franchise data
                    mock_openai_instance = Mock()
                    
                    franchise_result = Mock()
                    franchise_result.success = True
                    franchise_result.results = [
                        Mock(
                            media_item_id=movie_id,
                            category_id=1,
                            matches=True,
                            confidence=0.95,
                            reasoning="Star Wars Episode IV",
                            franchise_position=4,
                            franchise_year=0,
                            franchise_reasoning="Episode IV in chronological order",
                            error=None
                        )
                    ]
                    franchise_result.total_tokens = 500
                    franchise_result.total_cost = 0.05
                    franchise_result.processing_time = 1.0
                    
                    mock_openai_instance.classify_media_items.return_value = [franchise_result]
                    mock_openai.return_value = mock_openai_instance
                    
                    # Create category
                    with test_db_manager.get_session() as session:
                        from models.database_models import CollectionCategory
                        category = CollectionCategory(
                            name="Star Wars Saga",
                            description="Star Wars movies",
                            prompt="Is this a Star Wars movie?",
                            is_franchise=True,
                            chronological_sorting=True
                        )
                        session.add(category)
                        session.commit()
                    
                    # Create collection manager and run classification
                    collection_manager = CollectionManager()
                    
                    # Mock the library scan part to skip it
                    with patch.object(collection_manager, '_scan_libraries'):
                        await collection_manager._classify_media_items(
                            category_names=["Star Wars Saga"],
                            scan_stats={
                                "ai_requests_made": 0,
                                "total_tokens_used": 0,
                                "total_cost_estimate": 0.0,
                                "classifications_processed": 0,
                                "items_failed": 0
                            },
                            progress_callback=None
                        )
                    
                    # Verify franchise information was stored
                    with test_db_manager.get_session() as session:
                        from models.database_models import MediaItem
                        updated_movie = session.get(MediaItem, movie_id)
                        
                        assert updated_movie.franchise_name == "Star Wars Saga"
                        assert updated_movie.chronological_order == 4
                        assert updated_movie.franchise_year == 0
    
    def test_collection_creation_with_chronological_order(self, test_session, test_db_manager):
        """Test that Plex collections are created with proper chronological ordering."""
        
        # This test would verify that when collections are created in Plex,
        # franchise movies are ordered chronologically rather than by release date
        
        # Create franchise movies
        movies_data = [
            ("sw4", "Star Wars: Episode IV", 1977, 4),  # Released first, but 4th chronologically
            ("sw1", "Star Wars: Episode I", 1999, 1),  # Released later, but 1st chronologically
            ("sw5", "Star Wars: Episode V", 1980, 5),
        ]
        
        for plex_key, title, year, chrono_order in movies_data:
            test_db_manager.get_or_create_media_item(
                test_session,
                plex_key=plex_key,
                title=title,
                year=year,
                type="movie",
                library_section="Movies",
                franchise_name="Star Wars Saga",
                chronological_order=chrono_order
            )
        
        # Create category
        from models.database_models import CollectionCategory
        category = CollectionCategory(
            name="Star Wars Saga",
            description="Star Wars movies",
            is_franchise=True,
            chronological_sorting=True
        )
        test_session.add(category)
        test_session.flush()
        
        # Create classifications
        from models.database_models import ItemClassification, MediaItem
        movies = test_session.query(MediaItem).filter(
            MediaItem.franchise_name == "Star Wars Saga"
        ).all()
        
        for movie in movies:
            classification = ItemClassification(
                media_item_id=movie.id,
                category_id=category.id,
                matches=True
            )
            test_session.add(classification)
        
        test_session.commit()
        
        # Get items in the order they would be sent to Plex
        # (should be chronological, not release order)
        matching_items = test_db_manager.get_category_items(
            test_session, category.id, matches_only=True
        )
        
        # The collection manager should order these chronologically
        # when creating the Plex collection
        ordered_movies = test_db_manager.get_franchise_movies(
            test_session, "Star Wars Saga", ordered=True
        )
        
        # Verify chronological ordering
        chronological_order = [movie.chronological_order for movie in ordered_movies]
        assert chronological_order == [1, 4, 5]  # Episode I, IV, V
        
        # Verify this is different from release order
        release_order = sorted(ordered_movies, key=lambda x: x.year)
        release_years = [movie.year for movie in release_order]
        assert release_years == [1977, 1980, 1999]  # Different from chronological


class TestErrorHandling:
    """Test error handling in franchise system."""
    
    def test_missing_franchise_data_handling(self, test_session, test_db_manager):
        """Test handling of movies with missing franchise data."""
        
        # Create movie without franchise info
        movie, _ = test_db_manager.get_or_create_media_item(
            test_session,
            plex_key="incomplete_movie",
            title="Incomplete Movie",
            year=2000,
            type="movie"
            # No franchise_name, chronological_order, etc.
        )
        
        # This should not cause errors when querying franchise data
        franchise_movies = test_db_manager.get_franchise_movies(
            test_session, "Nonexistent Franchise", ordered=True
        )
        
        assert franchise_movies == []
        
        all_franchises = test_db_manager.get_all_franchises(test_session)
        assert all_franchises == {}
    
    def test_malformed_api_response_handling(self):
        """Test handling of malformed OpenAI API responses."""
        from api.openai_client import OpenAIClient
        
        with patch('api.openai_client.OpenAI'):
            client = OpenAIClient(api_key="test", model="test")
            
            # Test with invalid JSON
            mock_response = Mock()
            mock_response.choices[0].message.content = "Invalid JSON response"
            mock_response.usage.total_tokens = 100
            
            results = client._parse_mega_batch_response(
                mock_response, 
                [{"id": 1, "title": "Test Movie"}], 
                [{"id": 1, "name": "Test Category"}]
            )
            
            # Should return error results instead of crashing
            assert len(results) == 1
            assert results[0].error is not None
            assert "Parse error" in results[0].error
    
    def test_database_column_migration_safety(self, temp_db_path):
        """Test that missing database columns are handled gracefully."""
        from api.database import DatabaseManager
        
        # Create database manager - should handle missing columns gracefully
        db_manager = DatabaseManager(temp_db_path)
        
        # Try to use franchise features - should not crash even if columns don't exist
        with db_manager.get_session() as session:
            # These operations should work even if franchise columns are missing
            franchises = db_manager.get_all_franchises(session)
            assert isinstance(franchises, dict)
            
            # This should also work gracefully
            franchise_movies = db_manager.get_franchise_movies(session, "Test", ordered=True)
            assert isinstance(franchise_movies, list)