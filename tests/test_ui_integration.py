"""Tests for UI integration and collection management interface."""

import pytest
from unittest.mock import Mock, patch


class TestCategoryDisplay:
    """Test category display and management UI components."""

    def test_category_source_badge(self, test_db_manager):
        """Test that categories display correct source badges."""
        with test_db_manager.get_session() as session:
            # Create config category
            config_cat = test_db_manager.create_collection_category(
                session, "Config Category", "Desc", "Prompt", source="config"
            )
            # Create NL category
            nl_cat = test_db_manager.create_natural_language_collection(
                session, "NL Category", "movies about time travel"
            )

            categories = test_db_manager.get_collection_categories(session, enabled_only=False)

            sources = {cat.name: getattr(cat, 'source', 'config') for cat in categories}
            assert sources["Config Category"] == "config"
            assert sources["NL Category"] == "natural_language"

    def test_nl_category_shows_original_query(self, test_db_manager):
        """Test that NL categories display their original query."""
        with test_db_manager.get_session() as session:
            nl_cat = test_db_manager.create_natural_language_collection(
                session, "Time Travel", "movies about time travel and paradoxes"
            )

            assert nl_cat.natural_language_query == "movies about time travel and paradoxes"


class TestDatabaseOptimization:
    """Test database optimization features."""

    def test_database_initialization_idempotency(self, temp_db_path):
        """Test that database initialization can be called multiple times safely."""
        from api.database import DatabaseManager

        db_manager1 = DatabaseManager(temp_db_path)
        assert db_manager1._initialized is True

        db_manager2 = DatabaseManager(temp_db_path)
        assert db_manager2._initialized is True

        with db_manager1.get_session() as session1:
            pass

        with db_manager2.get_session() as session2:
            pass

    def test_migration_idempotency(self, test_db_manager):
        """Test that database migration can be run multiple times safely."""
        test_db_manager.migrate_database()
        test_db_manager.migrate_database()
        test_db_manager.migrate_database()

        assert test_db_manager._migration_completed is True

    def test_new_columns_exist_after_migration(self, test_db_manager):
        """Test that new source and NL columns are available after migration."""
        with test_db_manager.get_session() as session:
            # Should be able to create NL categories (uses source and natural_language_query columns)
            cat = test_db_manager.create_natural_language_collection(
                session, "Test", "test query"
            )
            assert cat.source == "natural_language"
            assert cat.natural_language_query == "test query"


class TestConfigBackwardCompatibility:
    """Test backward compatibility with old config format."""

    def test_ai_config_alias(self):
        """Test that AIConfig is importable as OpenAIConfig for backward compat."""
        from utils.config import AIConfig, OpenAIConfig
        assert AIConfig is OpenAIConfig

    def test_app_config_uses_ai_field(self):
        """Test that AppConfig uses 'ai' field instead of 'openai'."""
        from utils.config import AppConfig, PlexConfig, AIConfig

        config = AppConfig(
            plex=PlexConfig(server_url="http://localhost:32400", token="test"),
            ai=AIConfig(api_key="test_key", model="gpt-4o-mini")
        )

        assert config.ai.api_key == "test_key"
        assert config.ai.model == "gpt-4o-mini"
        assert config.ai.provider == "openai"

    def test_collection_category_no_franchise_field(self):
        """Test that CollectionCategory no longer requires franchise field."""
        from utils.config import CollectionCategory

        cat = CollectionCategory(
            name="Test",
            description="Test desc",
            prompt="Test prompt"
        )

        assert cat.name == "Test"
        # Should not have a franchise attribute
        assert not hasattr(cat, 'franchise')
