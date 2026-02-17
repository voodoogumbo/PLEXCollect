"""Database models for PLEXCollect."""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, 
    ForeignKey, JSON, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func

Base = declarative_base()

class MediaItem(Base):
    """Represents a media item (movie or TV show) from Plex."""
    
    __tablename__ = "media_items"
    
    id = Column(Integer, primary_key=True)
    plex_key = Column(String(255), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False, index=True)
    year = Column(Integer, nullable=True, index=True)
    type = Column(String(50), nullable=False, index=True)  # movie, show, episode
    summary = Column(Text, nullable=True)
    genres = Column(JSON, nullable=True)  # List of genre strings
    directors = Column(JSON, nullable=True)  # List of director names
    actors = Column(JSON, nullable=True)  # List of actor names
    studios = Column(JSON, nullable=True)  # List of studio names
    collections = Column(JSON, nullable=True)  # List of existing collection names
    content_rating = Column(String(20), nullable=True)
    duration = Column(Integer, nullable=True)  # Duration in minutes
    rating = Column(Float, nullable=True)  # IMDb/user rating
    library_section = Column(String(100), nullable=False, index=True)
    
    # Franchise information
    franchise_name = Column(String(200), nullable=True, index=True)  # Name of franchise (e.g., "Star Wars")
    chronological_order = Column(Integer, nullable=True, index=True)  # Position in franchise timeline
    franchise_year = Column(Integer, nullable=True)  # In-universe year for chronological sorting
    franchise_notes = Column(Text, nullable=True)  # Manual notes about franchise positioning
    manual_franchise_override = Column(Boolean, default=False)  # User manually set franchise info
    
    # Metadata
    added_at = Column(DateTime, nullable=True)
    updated_at = Column(DateTime, nullable=True)
    last_scanned = Column(DateTime, default=func.now(), index=True)
    
    # Relationships
    classifications = relationship("ItemClassification", back_populates="media_item", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<MediaItem(title='{self.title}', year={self.year}, type='{self.type}')>"

class CollectionCategory(Base):
    """Represents a collection category/rule for AI classification."""
    
    __tablename__ = "collection_categories"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    prompt = Column(Text, nullable=False)
    enabled = Column(Boolean, default=True, index=True)
    
    # Collection settings
    auto_create = Column(Boolean, default=True)
    update_existing = Column(Boolean, default=True)
    collection_mode = Column(String(50), default="default")  # default, hide, hideItems, showItems
    collection_order = Column(String(50), default="release")  # release, alpha, custom

    # Source tracking
    source = Column(String(50), default="config")  # "config" or "natural_language"
    natural_language_query = Column(Text, nullable=True)  # Original NL query if source is "natural_language"
    
    # Statistics
    item_count = Column(Integer, default=0)
    last_updated = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    classifications = relationship("ItemClassification", back_populates="category", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<CollectionCategory(name='{self.name}', enabled={self.enabled})>"

class ItemClassification(Base):
    """Represents the AI classification of a media item to a category."""
    
    __tablename__ = "item_classifications"
    
    id = Column(Integer, primary_key=True)
    media_item_id = Column(Integer, ForeignKey("media_items.id"), nullable=False)
    category_id = Column(Integer, ForeignKey("collection_categories.id"), nullable=False)
    
    # Classification results
    matches = Column(Boolean, nullable=False, index=True)
    confidence = Column(Float, nullable=True)  # AI confidence score (0-1)
    ai_reasoning = Column(Text, nullable=True)  # AI explanation for the classification
    
    # Processing metadata
    model_used = Column(String(100), nullable=True)
    tokens_used = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)  # Seconds
    classified_at = Column(DateTime, default=func.now(), index=True)
    
    # Franchise-specific information (for franchise collections)
    franchise_position = Column(Integer, nullable=True)  # AI-determined chronological position
    franchise_confidence = Column(Float, nullable=True)  # Confidence in franchise detection (0-1)
    franchise_year = Column(Integer, nullable=True)  # AI-determined in-universe year
    franchise_reasoning = Column(Text, nullable=True)  # AI explanation for franchise positioning
    
    # Status
    needs_review = Column(Boolean, default=False, index=True)
    manually_overridden = Column(Boolean, default=False)
    
    # Relationships
    media_item = relationship("MediaItem", back_populates="classifications")
    category = relationship("CollectionCategory", back_populates="classifications")
    
    # Ensure unique classification per item-category pair
    __table_args__ = (
        UniqueConstraint('media_item_id', 'category_id', name='unique_item_category_classification'),
        Index('idx_matches_category', 'matches', 'category_id'),
    )
    
    def __repr__(self):
        return f"<ItemClassification(item_id={self.media_item_id}, category_id={self.category_id}, matches={self.matches})>"

class ScanHistory(Base):
    """Tracks library scan history and statistics."""
    
    __tablename__ = "scan_history"
    
    id = Column(Integer, primary_key=True)
    started_at = Column(DateTime, default=func.now(), index=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    status = Column(String(50), nullable=False, index=True)  # running, completed, failed, cancelled
    
    # Scan scope
    library_sections = Column(JSON, nullable=True)  # List of library sections scanned
    scan_type = Column(String(50), default="full")  # full, incremental, category_only
    
    # Statistics
    items_scanned = Column(Integer, default=0)
    items_added = Column(Integer, default=0)
    items_updated = Column(Integer, default=0)
    items_removed = Column(Integer, default=0)
    classifications_processed = Column(Integer, default=0)
    collections_created = Column(Integer, default=0)
    collections_updated = Column(Integer, default=0)
    
    # Resource usage
    ai_requests_made = Column(Integer, default=0)
    total_tokens_used = Column(Integer, default=0)
    total_cost_estimate = Column(Float, default=0.0)
    
    # Error tracking
    error_message = Column(Text, nullable=True)
    items_failed = Column(Integer, default=0)
    
    def __repr__(self):
        return f"<ScanHistory(started_at='{self.started_at}', status='{self.status}', items_scanned={self.items_scanned})>"

class AIProcessingLog(Base):
    """Logs individual AI API calls for debugging and cost tracking."""
    
    __tablename__ = "ai_processing_logs"
    
    id = Column(Integer, primary_key=True)
    scan_id = Column(Integer, ForeignKey("scan_history.id"), nullable=True)
    
    # Request details
    model_used = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    
    # Response details
    items_in_batch = Column(Integer, nullable=False)
    processing_time = Column(Float, nullable=False)  # Seconds
    success = Column(Boolean, nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    
    # Cost estimation
    estimated_cost = Column(Float, nullable=True)
    
    # Timestamps
    requested_at = Column(DateTime, default=func.now(), index=True)
    
    def __repr__(self):
        return f"<AIProcessingLog(model='{self.model_used}', tokens={self.total_tokens}, success={self.success})>"

class UserFeedback(Base):
    """Stores user feedback on AI classifications for improving accuracy."""
    
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True)
    classification_id = Column(Integer, ForeignKey("item_classifications.id"), nullable=False)
    
    # Feedback
    user_agrees = Column(Boolean, nullable=False)
    user_comment = Column(Text, nullable=True)
    corrected_classification = Column(Boolean, nullable=True)  # What user thinks it should be
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    ip_address = Column(String(45), nullable=True)  # For analytics, not identification
    
    def __repr__(self):
        return f"<UserFeedback(classification_id={self.classification_id}, agrees={self.user_agrees})>"