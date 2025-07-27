"""Configuration management for PLEXCollect."""

import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
import logging

class PlexConfig(BaseModel):
    server_url: str = Field(..., description="Plex server URL")
    token: str = Field(..., description="Plex authentication token")
    library_sections: List[str] = Field(default_factory=list, description="Library sections to scan")

class RateLimitConfig(BaseModel):
    requests_per_minute: int = Field(default=20, description="Max requests per minute")
    tokens_per_minute: int = Field(default=40000, description="Max tokens per minute")

class OpenAIConfig(BaseModel):
    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4", description="Model to use")
    max_tokens: int = Field(default=500, description="Max tokens per response")
    temperature: float = Field(default=0.1, description="Model temperature")
    batch_size: int = Field(default=10, description="Batch size for processing")
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

class CollectionCategory(BaseModel):
    name: str = Field(..., description="Collection name")
    description: str = Field(..., description="Collection description")
    prompt: str = Field(..., description="AI classification prompt")
    franchise: bool = Field(default=False, description="Is this a franchise collection requiring chronological ordering?")

class CollectionsConfig(BaseModel):
    default_categories: List[CollectionCategory] = Field(default_factory=list)
    auto_create: bool = Field(default=True, description="Auto-create missing collections")
    update_existing: bool = Field(default=True, description="Update existing collections")
    remove_missing: bool = Field(default=False, description="Remove items that no longer match")

class DatabaseConfig(BaseModel):
    path: str = Field(default="data/collections.db", description="Database file path")
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")
    backup_interval_hours: int = Field(default=24, description="Hours between backups")

class SchedulingConfig(BaseModel):
    auto_scan_enabled: bool = Field(default=False, description="Enable automatic scanning")
    scan_interval_hours: int = Field(default=24, description="Hours between scans")
    scan_time: str = Field(default="02:00", description="Time to run scans (HH:MM)")

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", description="Logging level")
    file_enabled: bool = Field(default=True, description="Enable file logging")
    file_path: str = Field(default="data/plexcollect.log", description="Log file path")
    max_file_size_mb: int = Field(default=10, description="Max log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup log files")

class AppConfig(BaseModel):
    plex: PlexConfig
    openai: OpenAIConfig
    collections: CollectionsConfig = Field(default_factory=CollectionsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scheduling: SchedulingConfig = Field(default_factory=SchedulingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @validator('plex')
    def validate_plex_config(cls, v):
        if not v.server_url:
            raise ValueError("Plex server URL is required")
        if not v.token:
            raise ValueError("Plex token is required")
        return v

    @validator('openai')
    def validate_openai_config(cls, v):
        if not v.api_key:
            raise ValueError("OpenAI API key is required")
        return v

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Check for environment variable overrides
            self._apply_env_overrides(config_data)
            
            self._config = AppConfig(**config_data)
            return self._config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading config: {e}")
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> None:
        """Apply environment variable overrides to config."""
        
        # Plex overrides
        if os.getenv("PLEX_SERVER_URL"):
            config_data.setdefault("plex", {})["server_url"] = os.getenv("PLEX_SERVER_URL")
        if os.getenv("PLEX_TOKEN"):
            config_data.setdefault("plex", {})["token"] = os.getenv("PLEX_TOKEN")
        
        # OpenAI overrides
        if os.getenv("OPENAI_API_KEY"):
            config_data.setdefault("openai", {})["api_key"] = os.getenv("OPENAI_API_KEY")
        if os.getenv("OPENAI_MODEL"):
            config_data.setdefault("openai", {})["model"] = os.getenv("OPENAI_MODEL")
    
    def get_config(self) -> AppConfig:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from file."""
        self._config = None
        return self.load_config()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        try:
            config = self.get_config()
            
            # Test Plex connection
            if not config.plex.server_url.startswith(('http://', 'https://')):
                issues.append("Plex server URL must start with http:// or https://")
            
            # Validate OpenAI settings
            if config.openai.temperature < 0 or config.openai.temperature > 2:
                issues.append("OpenAI temperature must be between 0 and 2")
            
            if config.openai.batch_size < 1 or config.openai.batch_size > 100:
                issues.append("OpenAI batch size must be between 1 and 100")
            
            # Validate database path
            db_dir = Path(config.database.path).parent
            if not db_dir.exists():
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create database directory: {e}")
            
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues

# Global config manager instance
config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get the application configuration."""
    return config_manager.get_config()

def reload_config() -> AppConfig:
    """Reload the application configuration."""
    return config_manager.reload_config()