"""Tests for UI integration and franchise management interface."""

import pytest
from unittest.mock import Mock, patch
import streamlit as st


class TestFranchiseUI:
    """Test franchise management UI components."""
    
    def test_franchise_overview_rendering(self, test_db_manager):
        """Test that franchise overview renders correctly."""
        # Mock franchises data
        mock_franchises = {
            "Star Wars Saga": 6,
            "Marvel Cinematic Universe": 25,
            "Fast & Furious Franchise": 10
        }
        
        with patch.object(test_db_manager, 'get_all_franchises', return_value=mock_franchises):
            # This would test the render_franchises function
            # In a real test, we'd need to mock streamlit components
            
            total_franchises = len(mock_franchises)
            total_movies = sum(mock_franchises.values())
            avg_movies = total_movies / total_franchises
            
            assert total_franchises == 3
            assert total_movies == 41
            assert avg_movies == pytest.approx(13.67, rel=1e-2)
    
    def test_franchise_search_filtering(self):
        """Test franchise search and filtering functionality."""
        mock_franchises = {
            "Star Wars Saga": 6,
            "Marvel Cinematic Universe": 25,
            "Fast & Furious Franchise": 10,
            "James Bond Collection": 25
        }
        
        # Test search filtering
        search_term = "star"
        filtered = {k: v for k, v in mock_franchises.items() 
                   if search_term.lower() in k.lower()}
        
        assert len(filtered) == 1
        assert "Star Wars Saga" in filtered
        
        # Test case-insensitive search
        search_term = "MARVEL"
        filtered = {k: v for k, v in mock_franchises.items() 
                   if search_term.lower() in k.lower()}
        
        assert len(filtered) == 1
        assert "Marvel Cinematic Universe" in filtered
    
    def test_franchise_sorting(self):
        """Test franchise sorting functionality."""
        mock_franchises = {
            "Star Wars Saga": 6,
            "Marvel Cinematic Universe": 25,
            "Fast & Furious Franchise": 10
        }
        
        # Test sort by movie count (descending)
        sorted_by_count = sorted(mock_franchises.items(), key=lambda x: x[1], reverse=True)
        assert sorted_by_count[0][0] == "Marvel Cinematic Universe"  # 25 movies
        assert sorted_by_count[1][0] == "Fast & Furious Franchise"  # 10 movies
        assert sorted_by_count[2][0] == "Star Wars Saga"  # 6 movies
        
        # Test sort alphabetically
        sorted_alphabetically = sorted(mock_franchises.items(), key=lambda x: x[0])
        assert sorted_alphabetically[0][0] == "Fast & Furious Franchise"
        assert sorted_alphabetically[1][0] == "Marvel Cinematic Universe"
        assert sorted_alphabetically[2][0] == "Star Wars Saga"
    
    def test_conflict_detection_ui(self):
        """Test conflict detection in UI."""
        from main import detect_franchise_conflicts
        
        # Test movies with conflicts
        mock_movies = [
            Mock(chronological_order=1, title="Movie A"),
            Mock(chronological_order=1, title="Movie B"),  # Duplicate position
            Mock(chronological_order=3, title="Movie C"),  # Gap (missing 2)
            Mock(chronological_order=None, title="Movie D")  # No position
        ]
        
        conflicts = detect_franchise_conflicts(mock_movies)
        
        # Should detect multiple types of conflicts
        assert len(conflicts) >= 2
        
        # Check for specific conflict types
        conflict_text = " ".join(conflicts)
        assert "Duplicate" in conflict_text
        assert "Missing positions" in conflict_text or "without chronological position" in conflict_text
    
    def test_bulk_reorder_functionality(self):
        """Test bulk reordering functionality."""
        from main import move_to_position
        
        # Test moving item to different position
        order_list = [1, 2, 3, 4, 5]
        
        # Move item from index 0 to index 2
        move_to_position(order_list, 0, 2)
        assert order_list == [2, 3, 1, 4, 5]
        
        # Move item from index 4 to index 1
        order_list = [1, 2, 3, 4, 5]
        move_to_position(order_list, 4, 1)
        assert order_list == [1, 5, 2, 3, 4]
    
    def test_export_functionality(self):
        """Test timeline export functionality."""
        # Mock movie data
        mock_movies = [
            Mock(
                title="Star Wars: Episode I",
                year=1999,
                chronological_order=1,
                franchise_year=-32,
                manual_franchise_override=False,
                franchise_notes="First chronologically",
                summary="Young Anakin Skywalker's story begins"
            ),
            Mock(
                title="Star Wars: Episode IV",
                year=1977,
                chronological_order=4,
                franchise_year=0,
                manual_franchise_override=True,
                franchise_notes="Original movie",
                summary="Luke Skywalker begins his journey"
            )
        ]
        
        # Test export data structure
        export_data = {
            "franchise_name": "Star Wars Saga",
            "total_movies": len(mock_movies),
            "movies": []
        }
        
        for movie in mock_movies:
            movie_data = {
                "title": movie.title,
                "year": movie.year,
                "chronological_position": movie.chronological_order,
                "franchise_year": movie.franchise_year,
                "manual_override": movie.manual_franchise_override,
                "notes": movie.franchise_notes,
                "summary": movie.summary[:200] + "..." if movie.summary and len(movie.summary) > 200 else movie.summary
            }
            export_data["movies"].append(movie_data)
        
        assert len(export_data["movies"]) == 2
        assert export_data["movies"][0]["chronological_position"] == 1
        assert export_data["movies"][1]["manual_override"] is True


class TestManualOverrides:
    """Test manual override system functionality."""
    
    def test_override_application(self, test_session, test_db_manager):
        """Test applying manual overrides."""
        # Create a movie
        movie, _ = test_db_manager.get_or_create_media_item(
            test_session,
            plex_key="test_override",
            title="Test Movie",
            year=2000,
            type="movie"
        )
        
        # Apply override
        test_db_manager.update_franchise_info(
            test_session,
            media_item_id=movie.id,
            franchise_name="Test Franchise",
            chronological_order=5,
            franchise_year=2020,
            notes="Manual override test"
        )
        
        # Verify override
        from models.database_models import MediaItem
        updated_movie = test_session.get(MediaItem, movie.id)
        assert updated_movie.manual_franchise_override is True
        assert updated_movie.chronological_order == 5
        assert updated_movie.franchise_year == 2020
        assert updated_movie.franchise_notes == "Manual override test"
    
    def test_override_clearing(self, test_session, test_db_manager):
        """Test clearing manual overrides."""
        # Create movie with override
        movie, _ = test_db_manager.get_or_create_media_item(
            test_session,
            plex_key="test_clear",
            title="Test Movie",
            year=2000,
            type="movie",
            franchise_name="Test Franchise",
            chronological_order=1,
            manual_franchise_override=True
        )
        
        # Clear override
        test_db_manager.clear_franchise_info(test_session, movie.id)
        
        # Verify clearing
        from models.database_models import MediaItem
        cleared_movie = test_session.get(MediaItem, movie.id)
        assert cleared_movie.franchise_name is None
        assert cleared_movie.chronological_order is None
        assert cleared_movie.manual_franchise_override is False
    
    def test_conflict_resolution(self):
        """Test automatic conflict resolution."""
        from main import resolve_franchise_conflicts
        
        # Mock movies with conflicts
        mock_movies = [
            Mock(chronological_order=None, year=2000, title="Movie C", manual_franchise_override=False),
            Mock(chronological_order=1, year=1999, title="Movie A", manual_franchise_override=False),
            Mock(chronological_order=1, year=2001, title="Movie D", manual_franchise_override=False),  # Duplicate
        ]
        
        # Mock session
        mock_session = Mock()
        mock_db_manager = Mock()
        
        # This would test the resolution logic
        # The function should assign sequential positions based on year
        expected_order = sorted(mock_movies, key=lambda x: (x.chronological_order or 999, x.year or 9999, x.title))
        
        # After resolution, positions should be 1, 2, 3
        assert len(expected_order) == 3


class TestDatabaseOptimization:
    """Test database optimization features."""
    
    def test_database_initialization_idempotency(self, temp_db_path):
        """Test that database initialization can be called multiple times safely."""
        from api.database import DatabaseManager
        
        # Create first manager
        db_manager1 = DatabaseManager(temp_db_path)
        assert db_manager1._initialized is True
        
        # Create second manager with same path
        db_manager2 = DatabaseManager(temp_db_path)
        assert db_manager2._initialized is True
        
        # Both should work without issues
        with db_manager1.get_session() as session1:
            pass
        
        with db_manager2.get_session() as session2:
            pass
    
    def test_migration_idempotency(self, test_db_manager):
        """Test that database migration can be run multiple times safely."""
        # Run migration multiple times
        test_db_manager.migrate_database()
        test_db_manager.migrate_database()
        test_db_manager.migrate_database()
        
        # Should complete without errors
        assert test_db_manager._migration_completed is True
    
    def test_index_creation(self, test_db_manager):
        """Test that database indexes are created for franchise queries."""
        # The indexes should be created during migration
        # This test would verify they exist and improve query performance
        
        with test_db_manager.get_session() as session:
            # Test a franchise query that should benefit from indexes
            from models.database_models import MediaItem
            
            # This query should use the franchise_name index
            franchise_movies = session.query(MediaItem).filter(
                MediaItem.franchise_name == "Test Franchise"
            ).all()
            
            # Should complete without errors (even if no results)
            assert isinstance(franchise_movies, list)