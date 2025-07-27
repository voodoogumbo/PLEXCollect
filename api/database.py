"""Database operations for PLEXCollect."""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from contextlib import contextmanager
from sqlalchemy import create_engine, and_, or_, desc, func, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from models.database_models import (
    Base, MediaItem, CollectionCategory, ItemClassification, 
    ScanHistory, AIProcessingLog, UserFeedback
)
from utils.config import get_config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for PLEXCollect."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize database manager."""
        if db_path is None:
            config = get_config()
            db_path = config.database.path
        
        # Ensure database directory exists
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Create database engine
        self.db_path = db_path
        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=False,  # Set to True for SQL debugging
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=20,
            max_overflow=30
        )
        
        # Create session factory
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Initialization tracking
        self._initialized = False
        self._migration_completed = False
        
        # Create tables (lazy initialization)
        self._ensure_initialized()
    
    def _ensure_initialized(self) -> None:
        """Ensure database is initialized (lazy initialization)."""
        if not self._initialized:
            self.init_database()
    
    def init_database(self) -> None:
        """Initialize database tables."""
        if self._initialized:
            logger.debug("Database already initialized, skipping")
            return
            
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables initialized successfully")
            
            # Run database migrations for new columns
            self.migrate_database()
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def migrate_database(self) -> None:
        """Migrate database schema to add missing columns."""
        if self._migration_completed:
            logger.debug("Database migration already completed, skipping")
            return
            
        logger.info("Checking for database schema migrations...")
        
        try:
            with self.engine.connect() as conn:
                # Check and add columns for media_items table
                self._add_column_if_missing(conn, "media_items", "franchise_name", "VARCHAR(200)")
                self._add_column_if_missing(conn, "media_items", "chronological_order", "INTEGER")
                self._add_column_if_missing(conn, "media_items", "franchise_year", "INTEGER")
                self._add_column_if_missing(conn, "media_items", "franchise_notes", "TEXT")
                self._add_column_if_missing(conn, "media_items", "manual_franchise_override", "BOOLEAN DEFAULT 0")
                
                # Check and add columns for collection_categories table
                self._add_column_if_missing(conn, "collection_categories", "is_franchise", "BOOLEAN DEFAULT 0")
                self._add_column_if_missing(conn, "collection_categories", "chronological_sorting", "BOOLEAN DEFAULT 0")
                
                # Check and add columns for item_classifications table
                self._add_column_if_missing(conn, "item_classifications", "franchise_position", "INTEGER")
                self._add_column_if_missing(conn, "item_classifications", "franchise_confidence", "FLOAT")
                self._add_column_if_missing(conn, "item_classifications", "franchise_year", "INTEGER")
                self._add_column_if_missing(conn, "item_classifications", "franchise_reasoning", "TEXT")
                
                # Create indexes for better franchise query performance
                self._create_index_if_missing(conn, "idx_media_franchise_name", "media_items", ["franchise_name"])
                self._create_index_if_missing(conn, "idx_media_chronological_order", "media_items", ["chronological_order"])
                self._create_index_if_missing(conn, "idx_categories_franchise", "collection_categories", ["is_franchise"])
                self._create_index_if_missing(conn, "idx_classifications_franchise_pos", "item_classifications", ["franchise_position"])
                
                # Commit all changes
                conn.commit()
                logger.info("Database migration completed successfully")
                self._migration_completed = True
                
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            # Don't raise here - let the app continue even if migration fails
    
    def _add_column_if_missing(self, conn, table_name: str, column_name: str, column_type: str) -> None:
        """Add a column to a table if it doesn't already exist."""
        try:
            # Try to check if column exists by querying it
            result = conn.execute(text(f"SELECT {column_name} FROM {table_name} LIMIT 1"))
            logger.debug(f"Column {table_name}.{column_name} already exists")
        except Exception:
            # Column doesn't exist, so add it
            try:
                sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                conn.execute(text(sql))
                logger.info(f"Added column {table_name}.{column_name}")
            except Exception as e:
                logger.warning(f"Failed to add column {table_name}.{column_name}: {e}")
    
    def _create_index_if_missing(self, conn, index_name: str, table_name: str, columns: List[str]) -> None:
        """Create an index if it doesn't already exist."""
        try:
            # Check if index exists
            result = conn.execute(text(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index_name}'"))
            if result.fetchone():
                logger.debug(f"Index {index_name} already exists")
                return
            
            # Create the index
            columns_str = ", ".join(columns)
            sql = f"CREATE INDEX {index_name} ON {table_name} ({columns_str})"
            conn.execute(text(sql))
            logger.info(f"Created index {index_name} on {table_name}({columns_str})")
            
        except Exception as e:
            logger.warning(f"Failed to create index {index_name}: {e}")
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    # Media Item Operations
    
    def get_or_create_media_item(self, session: Session, plex_key: str, **kwargs) -> Tuple[MediaItem, bool]:
        """Get existing media item or create new one."""
        item = session.query(MediaItem).filter(MediaItem.plex_key == plex_key).first()
        created = False
        
        if item is None:
            item = MediaItem(plex_key=plex_key, **kwargs)
            session.add(item)
            session.flush()  # Get the ID
            created = True
            logger.debug(f"Created new media item: {item.title}")
        else:
            # Update existing item
            for key, value in kwargs.items():
                if hasattr(item, key):
                    setattr(item, key, value)
            item.last_scanned = datetime.now()
            logger.debug(f"Updated existing media item: {item.title}")
        
        return item, created
    
    def get_media_items(self, session: Session, 
                       library_section: Optional[str] = None,
                       media_type: Optional[str] = None,
                       limit: Optional[int] = None) -> List[MediaItem]:
        """Get media items with optional filtering."""
        query = session.query(MediaItem)
        
        if library_section:
            query = query.filter(MediaItem.library_section == library_section)
        
        if media_type:
            query = query.filter(MediaItem.type == media_type)
        
        query = query.order_by(MediaItem.title)
        
        if limit:
            query = query.limit(limit)
        
        return query.all()
    
    def get_unclassified_items(self, session: Session, category_id: int, include_errors: bool = True) -> List[MediaItem]:
        """Get media items that haven't been classified for a specific category."""
        from models.database_models import ItemClassification
        from datetime import datetime, timedelta
        
        base_query = session.query(MediaItem)
        
        if include_errors:
            # Include items with no classifications OR items with error classifications older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            # Items with no classifications at all
            no_classifications = ~MediaItem.classifications.any(ItemClassification.category_id == category_id)
            
            # Items with only error classifications that are old
            error_classifications = MediaItem.classifications.any(
                (ItemClassification.category_id == category_id) &
                (ItemClassification.ai_reasoning.like('%error%')) &
                (ItemClassification.classified_at < cutoff_time)
            )
            
            return base_query.filter(no_classifications | error_classifications).all()
        else:
            # Original behavior - only items with no classifications
            return base_query.filter(
                ~MediaItem.classifications.any(ItemClassification.category_id == category_id)
            ).all()
    
    def clear_category_classifications(self, session: Session, category_id: int) -> int:
        """Clear all existing classifications for a specific category. Returns count of cleared items."""
        from models.database_models import ItemClassification
        
        deleted_count = session.query(ItemClassification).filter(
            ItemClassification.category_id == category_id
        ).delete(synchronize_session=False)
        
        session.commit()
        return deleted_count
    
    # Collection Category Operations
    
    def create_collection_category(self, session: Session, name: str, description: str, 
                                 prompt: str, **kwargs) -> CollectionCategory:
        """Create a new collection category."""
        category = CollectionCategory(
            name=name,
            description=description,
            prompt=prompt,
            **kwargs
        )
        session.add(category)
        session.flush()
        logger.info(f"Created collection category: {name}")
        return category
    
    def get_collection_categories(self, session: Session, enabled_only: bool = True) -> List[CollectionCategory]:
        """Get all collection categories."""
        query = session.query(CollectionCategory)
        if enabled_only:
            query = query.filter(CollectionCategory.enabled == True)
        return query.order_by(CollectionCategory.name).all()
    
    def get_category_by_name(self, session: Session, name: str) -> Optional[CollectionCategory]:
        """Get collection category by name."""
        return session.query(CollectionCategory).filter(CollectionCategory.name == name).first()
    
    def update_category_stats(self, session: Session, category_id: int) -> None:
        """Update statistics for a collection category."""
        item_count = session.query(ItemClassification).filter(
            and_(
                ItemClassification.category_id == category_id,
                ItemClassification.matches == True
            )
        ).count()
        
        category = session.get(CollectionCategory, category_id)
        if category:
            category.item_count = item_count
            category.last_updated = datetime.now()
    
    # Classification Operations
    
    def create_classification(self, session: Session, media_item_id: int, category_id: int,
                            matches: bool, confidence: Optional[float] = None,
                            ai_reasoning: Optional[str] = None, **kwargs) -> ItemClassification:
        """Create a new item classification."""
        classification = ItemClassification(
            media_item_id=media_item_id,
            category_id=category_id,
            matches=matches,
            confidence=confidence,
            ai_reasoning=ai_reasoning,
            **kwargs
        )
        
        try:
            session.add(classification)
            session.flush()
            return classification
        except IntegrityError:
            session.rollback()
            # Update existing classification
            existing = session.query(ItemClassification).filter(
                and_(
                    ItemClassification.media_item_id == media_item_id,
                    ItemClassification.category_id == category_id
                )
            ).first()
            
            if existing:
                existing.matches = matches
                existing.confidence = confidence
                existing.ai_reasoning = ai_reasoning
                existing.classified_at = datetime.now()
                for key, value in kwargs.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                return existing
            else:
                raise
    
    def get_category_items(self, session: Session, category_id: int, 
                          matches_only: bool = True) -> List[MediaItem]:
        """Get all media items for a category."""
        query = session.query(MediaItem).join(ItemClassification).filter(
            ItemClassification.category_id == category_id
        )
        
        if matches_only:
            query = query.filter(ItemClassification.matches == True)
        
        return query.order_by(MediaItem.title).all()
    
    def get_items_needing_review(self, session: Session) -> List[ItemClassification]:
        """Get classifications that need manual review."""
        return session.query(ItemClassification).filter(
            ItemClassification.needs_review == True
        ).join(MediaItem).join(CollectionCategory).order_by(
            CollectionCategory.name, MediaItem.title
        ).all()
    
    # Scan History Operations
    
    def create_scan_history(self, session: Session, library_sections: List[str], 
                          scan_type: str = "full") -> ScanHistory:
        """Create a new scan history record."""
        scan = ScanHistory(
            library_sections=library_sections,
            scan_type=scan_type,
            status="running"
        )
        session.add(scan)
        session.flush()
        logger.info(f"Started new scan: {scan.id}")
        return scan
    
    def update_scan_status(self, session: Session, scan_id: int, status: str, 
                          error_message: Optional[str] = None, **stats) -> None:
        """Update scan status and statistics."""
        scan = session.get(ScanHistory, scan_id)
        if scan:
            scan.status = status
            if error_message:
                scan.error_message = error_message
            if status in ["completed", "failed", "cancelled"]:
                scan.completed_at = datetime.now()
            
            # Update statistics
            for key, value in stats.items():
                if hasattr(scan, key):
                    setattr(scan, key, value)
    
    def get_recent_scans(self, session: Session, limit: int = 10) -> List[ScanHistory]:
        """Get recent scan history."""
        return session.query(ScanHistory).order_by(
            desc(ScanHistory.started_at)
        ).limit(limit).all()
    
    # AI Processing Log Operations
    
    def log_ai_processing(self, session: Session, model_used: str, prompt_tokens: int,
                         completion_tokens: int, items_in_batch: int, 
                         processing_time: float, success: bool,
                         scan_id: Optional[int] = None, 
                         error_message: Optional[str] = None,
                         estimated_cost: Optional[float] = None) -> AIProcessingLog:
        """Log an AI processing request."""
        log_entry = AIProcessingLog(
            scan_id=scan_id,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            items_in_batch=items_in_batch,
            processing_time=processing_time,
            success=success,
            error_message=error_message,
            estimated_cost=estimated_cost
        )
        session.add(log_entry)
        return log_entry
    
    def get_ai_usage_stats(self, session: Session, 
                          start_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Get AI usage statistics."""
        query = session.query(AIProcessingLog)
        
        if start_date:
            query = query.filter(AIProcessingLog.requested_at >= start_date)
        
        logs = query.all()
        
        if not logs:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "success_rate": 0.0,
                "avg_processing_time": 0.0
            }
        
        total_requests = len(logs)
        successful_requests = sum(1 for log in logs if log.success)
        total_tokens = sum(log.total_tokens for log in logs)
        total_cost = sum(log.estimated_cost or 0 for log in logs)
        avg_processing_time = sum(log.processing_time for log in logs) / total_requests
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "avg_processing_time": avg_processing_time
        }
    
    # Utility Operations
    
    def cleanup_old_data(self, session: Session, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean up old data beyond the retention period."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up old scan history
        old_scans = session.query(ScanHistory).filter(
            ScanHistory.started_at < cutoff_date
        ).count()
        
        session.query(ScanHistory).filter(
            ScanHistory.started_at < cutoff_date
        ).delete()
        
        # Clean up old AI processing logs
        old_ai_logs = session.query(AIProcessingLog).filter(
            AIProcessingLog.requested_at < cutoff_date
        ).count()
        
        session.query(AIProcessingLog).filter(
            AIProcessingLog.requested_at < cutoff_date
        ).delete()
        
        logger.info(f"Cleaned up {old_scans} old scans and {old_ai_logs} AI logs")
        
        return {
            "scans_removed": old_scans,
            "ai_logs_removed": old_ai_logs
        }
    
    def get_database_stats(self, session: Session) -> Dict[str, int]:
        """Get database statistics."""
        stats = {
            "media_items": session.query(MediaItem).count(),
            "collection_categories": session.query(CollectionCategory).count(),
            "classifications": session.query(ItemClassification).count(),
            "scan_history": session.query(ScanHistory).count(),
            "ai_processing_logs": session.query(AIProcessingLog).count(),
            "user_feedback": session.query(UserFeedback).count()
        }
        
        # Try to get franchise items count (may fail if column doesn't exist yet)
        try:
            stats["franchise_items"] = session.query(MediaItem).filter(MediaItem.franchise_name.isnot(None)).count()
        except Exception:
            stats["franchise_items"] = 0
            
        return stats
    
    # Franchise-specific operations
    
    def get_franchise_movies(self, session: Session, franchise_name: str, 
                           ordered: bool = True) -> List[MediaItem]:
        """Get all movies in a franchise, optionally ordered chronologically."""
        try:
            query = session.query(MediaItem).filter(MediaItem.franchise_name == franchise_name)
            
            if ordered:
                query = query.order_by(MediaItem.chronological_order.asc().nullslast(), MediaItem.year.asc())
            else:
                query = query.order_by(MediaItem.title)
            
            return query.all()
        except Exception as e:
            logger.warning(f"Error getting franchise movies (column may not exist yet): {e}")
            return []
    
    def get_all_franchises(self, session: Session) -> Dict[str, int]:
        """Get all franchises and their movie counts."""
        try:
            franchises = {}
            franchise_data = session.query(
                MediaItem.franchise_name, 
                func.count(MediaItem.id)
            ).filter(
                MediaItem.franchise_name.isnot(None)
            ).group_by(MediaItem.franchise_name).all()
            
            for franchise_name, count in franchise_data:
                franchises[franchise_name] = count
            
            return franchises
        except Exception as e:
            logger.warning(f"Error getting franchises (column may not exist yet): {e}")
            return {}
    
    def update_franchise_info(self, session: Session, media_item_id: int, 
                            franchise_name: str, chronological_order: Optional[int] = None,
                            franchise_year: Optional[int] = None, 
                            notes: Optional[str] = None) -> None:
        """Update franchise information for a media item."""
        try:
            item = session.get(MediaItem, media_item_id)
            if item:
                item.franchise_name = franchise_name
                item.chronological_order = chronological_order
                item.franchise_year = franchise_year
                item.franchise_notes = notes
                item.manual_franchise_override = True
                logger.info(f"Updated franchise info for {item.title}: {franchise_name}")
        except Exception as e:
            logger.warning(f"Error updating franchise info (columns may not exist yet): {e}")
    
    def clear_franchise_info(self, session: Session, media_item_id: int) -> None:
        """Clear franchise information for a media item."""
        try:
            item = session.get(MediaItem, media_item_id)
            if item:
                item.franchise_name = None
                item.chronological_order = None
                item.franchise_year = None
                item.franchise_notes = None
                item.manual_franchise_override = False
                logger.info(f"Cleared franchise info for {item.title}")
        except Exception as e:
            logger.warning(f"Error clearing franchise info (columns may not exist yet): {e}")

# Global database manager instance with thread safety
_db_manager: Optional[DatabaseManager] = None
_db_manager_lock = None

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance (thread-safe singleton)."""
    global _db_manager, _db_manager_lock
    
    if _db_manager is None:
        # Lazy import to avoid circular imports
        import threading
        if _db_manager_lock is None:
            _db_manager_lock = threading.Lock()
        
        with _db_manager_lock:
            # Double-check locking pattern
            if _db_manager is None:
                _db_manager = DatabaseManager()
                logger.info("Initialized global database manager instance")
    
    return _db_manager

def init_database() -> None:
    """Initialize the database (idempotent operation)."""
    get_database_manager()._ensure_initialized()