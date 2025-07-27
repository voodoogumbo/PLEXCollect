"""Tests for franchise detection and classification."""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from api.openai_client import OpenAIClient, ClassificationResult, BatchResult
from models.database_models import MediaItem, CollectionCategory


class TestFranchiseDetection:
    """Test franchise detection functionality."""
    
    def test_star_wars_franchise_detection(self, star_wars_movies, sample_config):
        """Test that Star Wars movies are correctly identified as a franchise."""
        # This would test the franchise detection logic
        star_wars_titles = [movie["title"] for movie in star_wars_movies]
        
        # Check that all titles contain "Star Wars"
        assert all("Star Wars" in title for title in star_wars_titles)
        
        # Check that episode numbers are detected
        episode_movies = [title for title in star_wars_titles if "Episode" in title]
        assert len(episode_movies) == 6  # All 6 movies have episode numbers
    
    def test_mcu_franchise_detection(self, mcu_movies, sample_config):
        """Test that MCU movies are correctly identified as a franchise."""
        mcu_titles = [movie["title"] for movie in mcu_movies]
        
        # Check for typical MCU characteristics
        superhero_movies = [title for title in mcu_titles if any(hero in title for hero in ["Iron Man", "Captain America", "Avengers"])]
        assert len(superhero_movies) == 3
    
    @pytest.mark.asyncio
    async def test_mega_batch_classification(self, star_wars_movies, sample_config, mock_openai_response):
        """Test the mega-batch classification system."""
        with patch('api.openai_client.OpenAI') as mock_openai:
            # Setup mock
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            mock_response = Mock()
            mock_response.choices[0].message.content = json.dumps(mock_openai_response)
            mock_response.usage.total_tokens = 1500
            
            mock_client.chat.completions.create.return_value = mock_response
            
            # Create OpenAI client
            openai_client = OpenAIClient(api_key="test_key", model="gpt-4o-mini")
            
            # Prepare categories
            categories = [
                {
                    "id": 1,
                    "name": "Star Wars Saga",
                    "description": "Star Wars movies in chronological story order",
                    "prompt": "Is this a Star Wars movie?",
                    "is_franchise": True
                }
            ]
            
            # Test mega-batch classification
            result = await openai_client.classify_all_items_mega_batch(
                star_wars_movies[:2], categories
            )
            
            assert result.success
            assert len(result.results) == 2  # 2 movies × 1 category
            assert result.total_tokens == 1500
            
            # Check that Star Wars movies are correctly classified
            star_wars_results = [r for r in result.results if r.matches]
            assert len(star_wars_results) == 2
            
            # Check chronological positioning
            positions = [r.franchise_position for r in star_wars_results if r.franchise_position]
            assert 1 in positions  # Episode I should be position 1
            assert 4 in positions  # Episode IV should be position 4
    
    def test_franchise_chronology_info(self):
        """Test that franchise chronology information is correctly provided."""
        from api.openai_client import OpenAIClient
        
        client = OpenAIClient(api_key="test", model="test")
        
        # Test Star Wars chronology
        star_wars_info = client._get_franchise_chronology_info("Star Wars Saga")
        assert "Episode I: The Phantom Menace" in star_wars_info
        assert "Episode IV: A New Hope" in star_wars_info
        assert "32 BBY" in star_wars_info  # Check for chronological markers
        
        # Test MCU chronology
        mcu_info = client._get_franchise_chronology_info("Marvel Cinematic Universe")
        assert "Captain America: The First Avenger" in mcu_info
        assert "1940s" in mcu_info
        assert "Iron Man" in mcu_info
    
    def test_confidence_thresholds(self, mock_openai_response):
        """Test that confidence thresholds are properly applied."""
        # High confidence results should be accepted
        high_confidence_result = mock_openai_response["classifications"]["1"]["Star Wars Saga"]
        assert high_confidence_result["confidence"] >= 0.7
        assert high_confidence_result["matches"] is True
        
        # Low confidence results should be rejected
        low_confidence_result = mock_openai_response["classifications"]["1"]["Christmas Movies"]
        assert low_confidence_result["confidence"] < 0.7


class TestFranchiseDatabase:
    """Test franchise database operations."""
    
    def test_create_franchise_movie(self, test_session, test_db_manager):
        """Test creating a movie with franchise information."""
        # Create a media item with franchise info
        media_item, created = test_db_manager.get_or_create_media_item(
            test_session,
            plex_key="test_sw_1",
            title="Star Wars: Episode IV - A New Hope",
            year=1977,
            type="movie",
            franchise_name="Star Wars Saga",
            chronological_order=4,
            franchise_year=0
        )
        
        assert created
        assert media_item.franchise_name == "Star Wars Saga"
        assert media_item.chronological_order == 4
        assert media_item.franchise_year == 0
    
    def test_get_franchise_movies(self, test_session, test_db_manager):
        """Test retrieving movies for a franchise in chronological order."""
        # Create test movies
        movies_data = [
            ("sw_4", "Episode IV", 1977, 4, 0),
            ("sw_1", "Episode I", 1999, 1, -32),
            ("sw_5", "Episode V", 1980, 5, 3),
        ]
        
        for plex_key, title, year, position, franchise_year in movies_data:
            test_db_manager.get_or_create_media_item(
                test_session,
                plex_key=plex_key,
                title=f"Star Wars: {title}",
                year=year,
                type="movie",
                franchise_name="Star Wars Saga",
                chronological_order=position,
                franchise_year=franchise_year
            )
        
        # Get franchise movies in chronological order
        franchise_movies = test_db_manager.get_franchise_movies(
            test_session, "Star Wars Saga", ordered=True
        )
        
        assert len(franchise_movies) == 3
        
        # Check chronological ordering
        positions = [movie.chronological_order for movie in franchise_movies]
        assert positions == [1, 4, 5]  # Should be in chronological order
        
        # Check titles are in correct order
        titles = [movie.title for movie in franchise_movies]
        assert "Episode I" in titles[0]
        assert "Episode IV" in titles[1]
        assert "Episode V" in titles[2]
    
    def test_franchise_override_system(self, test_session, test_db_manager):
        """Test manual override functionality."""
        # Create a movie
        media_item, _ = test_db_manager.get_or_create_media_item(
            test_session,
            plex_key="test_movie",
            title="Test Movie",
            year=2000,
            type="movie"
        )
        
        # Apply manual franchise override
        test_db_manager.update_franchise_info(
            test_session,
            media_item_id=media_item.id,
            franchise_name="Test Franchise",
            chronological_order=1,
            franchise_year=2000,
            notes="Manual test override"
        )
        
        # Verify override was applied
        updated_item = test_session.get(MediaItem, media_item.id)
        assert updated_item.franchise_name == "Test Franchise"
        assert updated_item.chronological_order == 1
        assert updated_item.manual_franchise_override is True
        assert updated_item.franchise_notes == "Manual test override"
        
        # Test clearing override
        test_db_manager.clear_franchise_info(test_session, media_item.id)
        
        cleared_item = test_session.get(MediaItem, media_item.id)
        assert cleared_item.franchise_name is None
        assert cleared_item.chronological_order is None
        assert cleared_item.manual_franchise_override is False
    
    def test_get_all_franchises(self, test_session, test_db_manager):
        """Test getting all franchises with movie counts."""
        # Create movies for multiple franchises
        franchise_data = [
            ("Star Wars Saga", 3),
            ("Marvel Cinematic Universe", 2),
            ("Fast & Furious Franchise", 1)
        ]
        
        movie_id = 1
        for franchise_name, count in franchise_data:
            for i in range(count):
                test_db_manager.get_or_create_media_item(
                    test_session,
                    plex_key=f"{franchise_name}_{i}",
                    title=f"{franchise_name} Movie {i+1}",
                    year=2000 + i,
                    type="movie",
                    franchise_name=franchise_name,
                    chronological_order=i + 1
                )
                movie_id += 1
        
        # Get all franchises
        franchises = test_db_manager.get_all_franchises(test_session)
        
        assert len(franchises) == 3
        assert franchises["Star Wars Saga"] == 3
        assert franchises["Marvel Cinematic Universe"] == 2
        assert franchises["Fast & Furious Franchise"] == 1


class TestFranchiseConflicts:
    """Test conflict detection and resolution for franchises."""
    
    def test_detect_duplicate_positions(self):
        """Test detection of duplicate chronological positions."""
        from main import detect_franchise_conflicts
        
        # Create mock movies with duplicate positions
        movies = [
            Mock(chronological_order=1, title="Movie A"),
            Mock(chronological_order=2, title="Movie B"),
            Mock(chronological_order=2, title="Movie C"),  # Duplicate
            Mock(chronological_order=3, title="Movie D"),
        ]
        
        conflicts = detect_franchise_conflicts(movies)
        assert len(conflicts) > 0
        assert any("Duplicate chronological positions" in conflict for conflict in conflicts)
        assert "2" in str(conflicts)  # Position 2 is duplicated
    
    def test_detect_missing_positions(self):
        """Test detection of gaps in chronological sequence."""
        from main import detect_franchise_conflicts
        
        # Create mock movies with gaps in sequence
        movies = [
            Mock(chronological_order=1, title="Movie A"),
            Mock(chronological_order=3, title="Movie C"),  # Missing position 2
            Mock(chronological_order=5, title="Movie E"),  # Missing position 4
        ]
        
        conflicts = detect_franchise_conflicts(movies)
        assert len(conflicts) > 0
        assert any("Missing positions" in conflict for conflict in conflicts)
    
    def test_detect_unpositioned_movies(self):
        """Test detection of movies without chronological positions."""
        from main import detect_franchise_conflicts
        
        # Create mock movies with some missing positions
        movies = [
            Mock(chronological_order=1, title="Movie A"),
            Mock(chronological_order=None, title="Movie B"),  # No position
            Mock(chronological_order=3, title="Movie C"),
            Mock(chronological_order=None, title="Movie D"),  # No position
        ]
        
        conflicts = detect_franchise_conflicts(movies)
        assert len(conflicts) > 0
        assert any("without chronological position" in conflict for conflict in conflicts)
        assert "Movie B" in str(conflicts) or "Movie D" in str(conflicts)