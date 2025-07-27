"""Pytest configuration and fixtures for PLEXCollect tests."""

import pytest
import tempfile
import os
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database_models import Base
from api.database import DatabaseManager
from utils.config import AppConfig, PlexConfig, OpenAIConfig, CollectionsConfig, CollectionCategory


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
        openai=OpenAIConfig(
            api_key="test_api_key",
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.1,
            batch_size=10
        ),
        collections=CollectionsConfig(
            default_categories=[
                CollectionCategory(
                    name="Star Wars Saga",
                    description="Star Wars movies in chronological story order",
                    prompt="Is this a Star Wars movie? If yes, what is its chronological position?",
                    franchise=True
                ),
                CollectionCategory(
                    name="Marvel Cinematic Universe",
                    description="MCU movies in chronological timeline order",
                    prompt="Is this an MCU movie? If yes, where does it fit in the MCU timeline?",
                    franchise=True
                ),
                CollectionCategory(
                    name="Christmas Movies",
                    description="Movies with Christmas themes",
                    prompt="Is this a Christmas movie?",
                    franchise=False
                )
            ]
        )
    )


@pytest.fixture
def star_wars_movies():
    """Sample Star Wars movie data for testing."""
    return [
        {
            "id": 1,
            "title": "Star Wars: Episode IV - A New Hope",
            "year": 1977,
            "type": "movie",
            "summary": "Luke Skywalker joins forces with a Jedi Knight, a cocky pilot, a Wookiee and two droids to save the galaxy.",
            "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
            "directors": ["George Lucas"]
        },
        {
            "id": 2,
            "title": "Star Wars: Episode I - The Phantom Menace",
            "year": 1999,
            "type": "movie",
            "summary": "Two Jedi escape a hostile blockade to find allies and come across a young boy who may bring balance to the Force.",
            "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
            "directors": ["George Lucas"]
        },
        {
            "id": 3,
            "title": "Star Wars: Episode V - The Empire Strikes Back",
            "year": 1980,
            "type": "movie",
            "summary": "After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training.",
            "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
            "directors": ["Irvin Kershner"]
        },
        {
            "id": 4,
            "title": "Star Wars: Episode II - Attack of the Clones",
            "year": 2002,
            "type": "movie",
            "summary": "Ten years after initially meeting, Anakin Skywalker shares a forbidden romance with Padmé Amidala.",
            "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
            "directors": ["George Lucas"]
        },
        {
            "id": 5,
            "title": "Star Wars: Episode VI - Return of the Jedi",
            "year": 1983,
            "type": "movie",
            "summary": "After a daring mission to rescue Han Solo from Jabba the Hutt, the Rebels dispatch to Endor to destroy the second Death Star.",
            "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
            "directors": ["Richard Marquand"]
        },
        {
            "id": 6,
            "title": "Star Wars: Episode III - Revenge of the Sith",
            "year": 2005,
            "type": "movie",
            "summary": "Three years into the Clone Wars, the Jedi rescue Palpatine from Count Dooku. As Obi-Wan pursues a new threat, Anakin acts as a double agent.",
            "genres": ["Action", "Adventure", "Fantasy", "Sci-Fi"],
            "directors": ["George Lucas"]
        }
    ]


@pytest.fixture
def mcu_movies():
    """Sample MCU movie data for testing."""
    return [
        {
            "id": 7,
            "title": "Iron Man",
            "year": 2008,
            "type": "movie",
            "summary": "After being held captive in an Afghan cave, billionaire engineer Tony Stark creates a unique weaponized suit of armor to fight evil.",
            "genres": ["Action", "Adventure", "Sci-Fi"],
            "directors": ["Jon Favreau"]
        },
        {
            "id": 8,
            "title": "Captain America: The First Avenger",
            "year": 2011,
            "type": "movie",
            "summary": "Steve Rogers, a rejected military soldier, transforms into Captain America after taking a dose of a 'Super-Soldier serum'.",
            "genres": ["Action", "Adventure", "Sci-Fi"],
            "directors": ["Joe Johnston"]
        },
        {
            "id": 9,
            "title": "The Avengers",
            "year": 2012,
            "type": "movie",
            "summary": "Earth's mightiest heroes must come together and learn to fight as a team if they are going to stop the mischievous Loki and his alien army from enslaving humanity.",
            "genres": ["Action", "Adventure", "Sci-Fi"],
            "directors": ["Joss Whedon"]
        }
    ]


@pytest.fixture
def expected_star_wars_chronology():
    """Expected chronological order for Star Wars movies."""
    return {
        "Star Wars: Episode I - The Phantom Menace": {"position": 1, "year": 32},  # 32 BBY
        "Star Wars: Episode II - Attack of the Clones": {"position": 2, "year": 22},  # 22 BBY
        "Star Wars: Episode III - Revenge of the Sith": {"position": 3, "year": 19},  # 19 BBY
        "Star Wars: Episode IV - A New Hope": {"position": 4, "year": 0},  # 0 ABY
        "Star Wars: Episode V - The Empire Strikes Back": {"position": 5, "year": 3},  # 3 ABY
        "Star Wars: Episode VI - Return of the Jedi": {"position": 6, "year": 4}  # 4 ABY
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response for testing."""
    return {
        "classifications": {
            "1": {
                "Star Wars Saga": {
                    "matches": True,
                    "confidence": 0.95,
                    "chronological_position": 4,
                    "franchise_year": 0,
                    "reasoning": "This is Episode IV, the first movie chronologically in the original trilogy"
                },
                "Marvel Cinematic Universe": {
                    "matches": False,
                    "confidence": 0.1,
                    "reasoning": "Not part of the MCU"
                },
                "Christmas Movies": {
                    "matches": False,
                    "confidence": 0.05,
                    "reasoning": "No Christmas themes"
                }
            },
            "2": {
                "Star Wars Saga": {
                    "matches": True,
                    "confidence": 0.95,
                    "chronological_position": 1,
                    "franchise_year": -32,
                    "reasoning": "This is Episode I, the first movie chronologically in the prequel trilogy"
                },
                "Marvel Cinematic Universe": {
                    "matches": False,
                    "confidence": 0.1,
                    "reasoning": "Not part of the MCU"
                },
                "Christmas Movies": {
                    "matches": False,
                    "confidence": 0.05,
                    "reasoning": "No Christmas themes"
                }
            }
        }
    }