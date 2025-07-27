"""Collection management system for PLEXCollect."""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Tuple
from sqlalchemy.orm import Session

from utils.config import get_config
from api.database import get_database_manager
from api.plex_client import get_plex_client
from api.openai_client import get_openai_client, ClassificationResult, BatchResult
from models.database_models import (
    MediaItem, CollectionCategory, ItemClassification, 
    ScanHistory, AIProcessingLog
)

logger = logging.getLogger(__name__)

class CollectionManager:
    """Manages the entire collection creation and update process."""
    
    def __init__(self):
        """Initialize collection manager."""
        self.config = get_config()
        self.db_manager = get_database_manager()
        self.plex_client = get_plex_client()
        self.openai_client = get_openai_client()
        
        # Current scan tracking
        self.current_scan_id: Optional[int] = None
        self.is_scanning = False
    
    async def full_scan_and_classify(self, 
                                   library_sections: Optional[List[str]] = None,
                                   categories: Optional[List[str]] = None,
                                   force_reclassify: bool = False,
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Perform a full scan and classification of the Plex library."""
        
        if self.is_scanning:
            raise RuntimeError("Scan already in progress")
        
        self.is_scanning = True
        scan_stats = {
            "items_scanned": 0,
            "items_added": 0,
            "items_updated": 0,
            "classifications_processed": 0,
            "collections_created": 0,
            "collections_updated": 0,
            "ai_requests_made": 0,
            "total_tokens_used": 0,
            "total_cost_estimate": 0.0,
            "items_failed": 0,
            "started_at": datetime.now(),
            "status": "running"
        }
        
        try:
            # Connect to Plex
            if not self.plex_client.connect():
                raise RuntimeError("Failed to connect to Plex server")
            
            # Start scan record
            with self.db_manager.get_session() as session:
                scan_sections = library_sections or [s.title for s in self.plex_client.get_library_sections()]
                scan_history = self.db_manager.create_scan_history(session, scan_sections)
                self.current_scan_id = scan_history.id
                scan_stats["scan_id"] = scan_history.id
            
            if progress_callback:
                progress_callback(0.1, "Connected to Plex server")
            
            # Phase 1: Scan library and update database
            logger.info("Phase 1: Scanning Plex library")
            await self._scan_libraries(library_sections, scan_stats, progress_callback)
            
            # Phase 2: Classify media items
            logger.info("Phase 2: Classifying media items with AI")
            if force_reclassify:
                logger.info("Force re-classification enabled - clearing existing classifications")
                if progress_callback:
                    progress_callback(0.35, "Clearing existing classifications for re-scan...")
                await self._clear_classifications_for_categories(categories)
            
            await self._classify_media_items(categories, scan_stats, progress_callback)
            
            # Phase 3: Update Plex collections
            logger.info("Phase 3: Updating Plex collections")
            await self._update_plex_collections(scan_stats, progress_callback)
            
            # Complete scan
            scan_stats["status"] = "completed"
            scan_stats["completed_at"] = datetime.now()
            
            with self.db_manager.get_session() as session:
                # Remove conflicting keys before passing to update_scan_status
                stats_copy = scan_stats.copy()
                stats_copy.pop('status', None)  # Remove status key to avoid conflict
                stats_copy.pop('scan_id', None)  # Remove scan_id key to avoid conflict
                
                self.db_manager.update_scan_status(
                    session, self.current_scan_id, "completed", **stats_copy
                )
            
            logger.info(f"Scan completed successfully: {scan_stats}")
            
        except Exception as e:
            logger.error(f"Scan failed: {e}")
            scan_stats["status"] = "failed"
            scan_stats["error_message"] = str(e)
            
            if self.current_scan_id:
                with self.db_manager.get_session() as session:
                    # Remove conflicting keys before passing to update_scan_status
                    stats_copy = scan_stats.copy()
                    stats_copy.pop('status', None)  # Remove status key to avoid conflict
                    stats_copy.pop('error_message', None)  # Remove error_message key to avoid conflict
                    stats_copy.pop('scan_id', None)  # Remove scan_id key to avoid conflict
                    
                    self.db_manager.update_scan_status(
                        session, self.current_scan_id, "failed", 
                        error_message=str(e), **stats_copy
                    )
            
            raise
        
        finally:
            self.is_scanning = False
            self.current_scan_id = None
            
            if progress_callback:
                if scan_stats["status"] == "completed":
                    progress_callback(1.0, "Scan completed successfully")
                else:
                    progress_callback(1.0, f"Scan failed: {scan_stats.get('error_message', 'Unknown error')}")
        
        return scan_stats
    
    async def _scan_libraries(self, library_sections: Optional[List[str]], 
                            scan_stats: Dict[str, Any], 
                            progress_callback: Optional[Callable]) -> None:
        """Scan Plex libraries and update database."""
        
        # Get target library sections
        if library_sections:
            sections = [s for s in self.plex_client.get_library_sections() 
                       if s.title in library_sections]
        else:
            # Use configured sections or all movie/show sections
            config_sections = self.config.plex.library_sections
            if config_sections:
                sections = [s for s in self.plex_client.get_library_sections() 
                           if s.title in config_sections]
            else:
                sections = [s for s in self.plex_client.get_library_sections() 
                           if s.type in ['movie', 'show']]
        
        if not sections:
            raise ValueError("No valid library sections found to scan")
        
        logger.info(f"Scanning {len(sections)} library sections")
        
        total_progress = 0.3  # This phase takes 30% of total progress
        section_progress = total_progress / len(sections)
        current_progress = 0.1  # Start after connection phase
        
        for section_idx, section in enumerate(sections):
            logger.info(f"Scanning section: {section.title}")
            
            if progress_callback:
                progress_callback(current_progress, f"Scanning library: {section.title}")
            
            items_in_section = 0
            
            try:
                # Scan this section
                with self.db_manager.get_session() as session:
                    for media_data in self.plex_client.scan_library_section(section.title):
                        try:
                            # Get or create media item
                            item, created = self.db_manager.get_or_create_media_item(
                                session, **media_data
                            )
                            
                            if created:
                                scan_stats["items_added"] += 1
                            else:
                                scan_stats["items_updated"] += 1
                            
                            scan_stats["items_scanned"] += 1
                            items_in_section += 1
                            
                            # Update progress periodically
                            if items_in_section % 50 == 0 and progress_callback:
                                section_progress_current = current_progress + (section_progress * 0.8)
                                progress_callback(
                                    section_progress_current, 
                                    f"Scanning {section.title}: {items_in_section} items"
                                )
                        
                        except Exception as e:
                            logger.warning(f"Error processing media item: {e}")
                            scan_stats["items_failed"] += 1
                            continue
                
                logger.info(f"Completed section {section.title}: {items_in_section} items")
                
            except Exception as e:
                logger.error(f"Error scanning section {section.title}: {e}")
                scan_stats["items_failed"] += items_in_section
                
            current_progress += section_progress
        
        logger.info(f"Library scan complete: {scan_stats['items_scanned']} items scanned")
    
    async def _clear_classifications_for_categories(self, category_names: Optional[List[str]]) -> None:
        """Clear existing classifications for specified categories to force re-classification."""
        
        with self.db_manager.get_session() as session:
            # Get categories to clear
            if category_names:
                categories = [self.db_manager.get_category_by_name(session, name) 
                             for name in category_names]
                categories = [c for c in categories if c is not None]
            else:
                categories = self.db_manager.get_collection_categories(session, enabled_only=True)
            
            if not categories:
                logger.warning("No categories found for clearing classifications")
                return
            
            # Clear classifications for each category
            total_cleared = 0
            for category in categories:
                cleared_count = self.db_manager.clear_category_classifications(session, category.id)
                total_cleared += cleared_count
                logger.info(f"Cleared {cleared_count} existing classifications for category: {category.name}")
            
            logger.info(f"Total classifications cleared: {total_cleared}")
    
    async def _classify_media_items(self, category_names: Optional[List[str]], 
                                  scan_stats: Dict[str, Any], 
                                  progress_callback: Optional[Callable]) -> None:
        """Classify media items using AI with mega-batch optimization."""
        
        with self.db_manager.get_session() as session:
            # Get categories to process
            if category_names:
                categories = [self.db_manager.get_category_by_name(session, name) 
                             for name in category_names]
                categories = [c for c in categories if c is not None]
            else:
                categories = self.db_manager.get_collection_categories(session, enabled_only=True)
            
            if not categories:
                logger.warning("No categories found for classification")
                return
            
            logger.info(f"Processing {len(categories)} categories with mega-batch optimization")
            
            # Get all media items that need classification (any category)
            all_unclassified_items = set()
            for category in categories:
                unclassified = self.db_manager.get_unclassified_items(session, category.id)
                all_unclassified_items.update(unclassified)
            
            all_unclassified_items = list(all_unclassified_items)
            
            if not all_unclassified_items:
                logger.info("No unclassified items found")
                return
            
            logger.info(f"Found {len(all_unclassified_items)} items needing classification across all categories")
            
            # Convert media items to format expected by OpenAI client
            media_data = []
            for item in all_unclassified_items:
                media_data.append({
                    'id': item.id,
                    'title': item.title,
                    'year': item.year,
                    'type': item.type,
                    'summary': item.summary or '',
                    'genres': item.genres or [],
                    'directors': item.directors or [],
                    'actors': item.actors or [],
                    'content_rating': item.content_rating,
                    'rating': item.rating
                })
            
            # Prepare all category data
            categories_data = []
            for category in categories:
                categories_data.append({
                    'id': category.id,
                    'name': category.name,
                    'description': category.description,
                    'prompt': category.prompt,
                    'is_franchise': category.is_franchise,
                    'chronological_sorting': category.chronological_sorting
                })
            
            total_progress = 0.5  # This phase takes 50% of total progress
            current_progress = 0.4  # Start after library scan
            
            if progress_callback:
                progress_callback(current_progress, "Starting mega-batch classification...")
            
            try:
                # Use mega-batch classification for all items across all categories
                batch_results = await self.openai_client.classify_media_items(
                    media_data, 
                    categories_data,
                    progress_callback=lambda p, msg: progress_callback(
                        current_progress + total_progress * p,
                        msg
                    ) if progress_callback else None
                )
                
                # Process results (should be just one mega-batch result)
                for batch_result in batch_results:
                    scan_stats["ai_requests_made"] += 1
                    scan_stats["total_tokens_used"] += batch_result.total_tokens
                    scan_stats["total_cost_estimate"] += batch_result.total_cost
                    
                    # Log AI processing
                    self.db_manager.log_ai_processing(
                        session,
                        model_used=self.openai_client.model,
                        prompt_tokens=batch_result.total_tokens // 2,  # Estimate
                        completion_tokens=batch_result.total_tokens // 2,
                        items_in_batch=len(batch_result.results),
                        processing_time=batch_result.processing_time,
                        success=batch_result.success,
                        scan_id=self.current_scan_id,
                        error_message=batch_result.error_message,
                        estimated_cost=batch_result.total_cost
                    )
                    
                    # Store all classifications
                    for result in batch_result.results:
                        try:
                            # Prepare franchise-specific kwargs
                            franchise_kwargs = {}
                            if result.franchise_position is not None:
                                franchise_kwargs['franchise_position'] = result.franchise_position
                            if result.franchise_year is not None:
                                franchise_kwargs['franchise_year'] = result.franchise_year
                            if result.franchise_reasoning is not None:
                                franchise_kwargs['franchise_reasoning'] = result.franchise_reasoning
                            
                            classification = self.db_manager.create_classification(
                                session,
                                media_item_id=result.media_item_id,
                                category_id=result.category_id,
                                matches=result.matches,
                                confidence=result.confidence,
                                ai_reasoning=result.reasoning,
                                model_used=self.openai_client.model,
                                tokens_used=result.tokens_used,
                                processing_time=result.processing_time,
                                **franchise_kwargs
                            )
                            scan_stats["classifications_processed"] += 1
                            
                            # Update franchise info on media item if it's a franchise match
                            if result.matches and result.franchise_position is not None:
                                # Find the category to get franchise name
                                category = next((c for c in categories if c.id == result.category_id), None)
                                if category and category.is_franchise:
                                    try:
                                        self.db_manager.update_franchise_info(
                                            session,
                                            media_item_id=result.media_item_id,
                                            franchise_name=category.name,
                                            chronological_order=result.franchise_position,
                                            franchise_year=result.franchise_year,
                                            notes=result.franchise_reasoning
                                        )
                                    except Exception as franchise_error:
                                        logger.warning(f"Could not update franchise info: {franchise_error}")
                            
                        except Exception as e:
                            logger.error(f"Error storing classification: {e}")
                            scan_stats["items_failed"] += 1
                
                # Update statistics for all categories
                for category in categories:
                    self.db_manager.update_category_stats(session, category.id)
                
                logger.info(f"Mega-batch classification complete: {scan_stats['classifications_processed']} classifications processed")
                logger.info(f"Total cost: ${scan_stats['total_cost_estimate']:.4f} (vs estimated ${len(all_unclassified_items) * len(categories) * 0.04:.2f} with old method)")
                
            except Exception as e:
                logger.error(f"Error in mega-batch classification: {e}")
                scan_stats["items_failed"] += len(all_unclassified_items) * len(categories)
        
        logger.info(f"Classification complete: {scan_stats['classifications_processed']} items classified")
    
    async def _update_plex_collections(self, scan_stats: Dict[str, Any], 
                                     progress_callback: Optional[Callable]) -> None:
        """Update Plex collections based on classifications."""
        
        current_progress = 0.9  # Start at 90%
        
        with self.db_manager.get_session() as session:
            categories = self.db_manager.get_collection_categories(session, enabled_only=True)
            
            if not categories:
                return
            
            progress_per_category = 0.1 / len(categories)
            
            for category in categories:
                if progress_callback:
                    progress_callback(
                        current_progress, 
                        f"Updating collection: {category.name}"
                    )
                
                try:
                    # Get items that match this category
                    matching_items = self.db_manager.get_category_items(
                        session, category.id, matches_only=True
                    )
                    
                    if not matching_items:
                        logger.info(f"No matching items for collection: {category.name}")
                        current_progress += progress_per_category
                        continue
                    
                    # Sort items for franchise collections using chronological order
                    if category.is_franchise and category.chronological_sorting:
                        logger.info(f"Sorting franchise collection '{category.name}' chronologically")
                        
                        # Get franchise items in chronological order
                        franchise_items = self.db_manager.get_franchise_movies(
                            session, category.name, ordered=True
                        )
                        
                        # Filter to only items that actually match this category
                        matching_item_ids = {item.id for item in matching_items}
                        matching_items = [item for item in franchise_items if item.id in matching_item_ids]
                        
                        logger.info(f"Ordered {len(matching_items)} franchise items chronologically")
                    
                    # Group by library section while preserving order
                    items_by_section: Dict[str, List[str]] = {}
                    for item in matching_items:
                        section = item.library_section
                        if section not in items_by_section:
                            items_by_section[section] = []
                        items_by_section[section].append(item.plex_key)
                    
                    # Update collections in each library section
                    for section, item_keys in items_by_section.items():
                        logger.info(f"Updating collection '{category.name}' in {section} with {len(item_keys)} items")
                        
                        try:
                            if self.plex_client.collection_exists(section, category.name):
                                # Update existing collection
                                success = self.plex_client.update_collection(
                                    section, category.name, item_keys
                                )
                                if success:
                                    scan_stats["collections_updated"] += 1
                                    logger.info(f"Updated collection '{category.name}' in {section}")
                            else:
                                # Create new collection
                                if self.config.collections.auto_create:
                                    success = self.plex_client.create_collection(
                                        section, category.name, item_keys, category.description
                                    )
                                    if success:
                                        scan_stats["collections_created"] += 1
                                        logger.info(f"Created collection '{category.name}' in {section}")
                                else:
                                    logger.info(f"Skipping creation of collection '{category.name}' (auto_create disabled)")
                        
                        except Exception as e:
                            logger.error(f"Error updating collection '{category.name}' in {section}: {e}")
                            scan_stats["items_failed"] += len(item_keys)
                
                except Exception as e:
                    logger.error(f"Error processing category {category.name}: {e}")
                
                current_progress += progress_per_category
        
        logger.info(f"Collection update complete: {scan_stats['collections_created']} created, "
                   f"{scan_stats['collections_updated']} updated")
    
    def get_scan_status(self) -> Dict[str, Any]:
        """Get current scan status."""
        return {
            "is_scanning": self.is_scanning,
            "current_scan_id": self.current_scan_id
        }
    
    def cancel_scan(self) -> bool:
        """Cancel the current scan."""
        if not self.is_scanning:
            return False
        
        # Note: This is a simple implementation
        # In a production system, you'd want more sophisticated cancellation
        self.is_scanning = False
        
        if self.current_scan_id:
            with self.db_manager.get_session() as session:
                self.db_manager.update_scan_status(
                    session, self.current_scan_id, "cancelled"
                )
        
        logger.info("Scan cancelled by user")
        return True
    
    def setup_default_categories(self) -> None:
        """Set up default collection categories from config."""
        with self.db_manager.get_session() as session:
            for category_config in self.config.collections.default_categories:
                existing = self.db_manager.get_category_by_name(session, category_config.name)
                if not existing:
                    self.db_manager.create_collection_category(
                        session,
                        name=category_config.name,
                        description=category_config.description,
                        prompt=category_config.prompt
                    )
                    logger.info(f"Created default category: {category_config.name}")

# Global collection manager instance
_collection_manager: Optional[CollectionManager] = None

def get_collection_manager() -> CollectionManager:
    """Get the global collection manager instance."""
    global _collection_manager
    if _collection_manager is None:
        _collection_manager = CollectionManager()
    return _collection_manager