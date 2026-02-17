"""Pytest configuration and fixtures for PLEXCollect tests."""

import pytest
import tempfile
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database_models import Base
from api.database import DatabaseManager
from utils.config import AppConfig, PlexConfig, AIConfig, CollectionsConfig, CollectionCategory


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_db_manager(temp_db_path):
    """Create a test database manager with temporary database."""
    db_manager = DatabaseManager(temp_db_path)
    return db_manager


@pytest.fixture
def test_session(test_db_manager):
    """Create a test database session."""
    with test_db_manager.get_session() as session:
        yield session


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return AppConfig(
        plex=PlexConfig(
            server_url="http://localhost:32400",
            token="test_token",
            library_sections=["Movies"]
        ),
        ai=AIConfig(
            provider="openai",
            api_key="test_api_key",
            model="gpt-4o-mini",
            max_tokens=4000,
            temperature=0.3,
            batch_size=10
        ),
        collections=CollectionsConfig(
            default_categories=[
                CollectionCategory(
                    name="Cozy Comfort Movies",
                    description="Warm, comforting movies perfect for a blanket-and-tea evening",
                    prompt="Does this movie have a cozy, comforting vibe?"
                ),
                CollectionCategory(
                    name="Mind-Bending Movies",
                    description="Movies that make you question reality",
                    prompt="Is this a mind-bending movie?"
                ),
                CollectionCategory(
                    name="Adrenaline Rush",
                    description="Non-stop action and intensity",
                    prompt="Is this a high-adrenaline movie?"
                )
            ]
        )
    )


@pytest.fixture
def sample_movies():
    """Sample movie data for testing vibe-based classification."""
    return [
        {
            "id": 1,
            "title": "Paddington",
            "year": 2014,
            "type": "movie",
            "summary": "A young Peruvian bear travels to London in search of a home.",
            "genres": ["Adventure", "Comedy", "Family"],
            "directors": ["Paul King"]
        },
        {
            "id": 2,
            "title": "Inception",
            "year": 2010,
            "type": "movie",
            "summary": "A thief who steals corporate secrets through dream-sharing technology.",
            "genres": ["Action", "Adventure", "Sci-Fi"],
            "directors": ["Christopher Nolan"]
        },
        {
            "id": 3,
            "title": "John Wick",
            "year": 2014,
            "type": "movie",
            "summary": "An ex-hit-man comes out of retirement to track down the gangsters that killed his dog.",
            "genres": ["Action", "Crime", "Thriller"],
            "directors": ["Chad Stahelski"]
        },
        {
            "id": 4,
            "title": "The Holiday",
            "year": 2006,
            "type": "movie",
            "summary": "Two women troubled with guy-Loss swap homes in each other's countries.",
            "genres": ["Comedy", "Romance"],
            "directors": ["Nancy Meyers"]
        },
        {
            "id": 5,
            "title": "Memento",
            "year": 2000,
            "type": "movie",
            "summary": "A man with short-term memory loss attempts to track down his wife's murderer.",
            "genres": ["Mystery", "Thriller"],
            "directors": ["Christopher Nolan"]
        },
        {
            "id": 6,
            "title": "Mad Max: Fury Road",
            "year": 2015,
            "type": "movie",
            "summary": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler.",
            "genres": ["Action", "Adventure", "Sci-Fi"],
            "directors": ["George Miller"]
        }
    ]


@pytest.fixture
def sample_nl_query():
    """Sample natural language query for testing."""
    return "movies about found family and belonging"


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing (index-based format)."""
    return {
        "Cozy Comfort Movies": [1, 4],
        "Mind-Bending Movies": [2, 5],
        "Adrenaline Rush": [3, 6]
    }
