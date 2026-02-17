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
                stats_copy = scan_stats.copy()
                stats_copy.pop('status', None)
                stats_copy.pop('scan_id', None)

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
                    stats_copy = scan_stats.copy()
                    stats_copy.pop('status', None)
                    stats_copy.pop('error_message', None)
                    stats_copy.pop('scan_id', None)

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

        if library_sections:
            sections = [s for s in self.plex_client.get_library_sections()
                       if s.title in library_sections]
        else:
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

        total_progress = 0.3
        section_progress = total_progress / len(sections)
        current_progress = 0.1

        for section_idx, section in enumerate(sections):
            logger.info(f"Scanning section: {section.title}")

            if progress_callback:
                progress_callback(current_progress, f"Scanning library: {section.title}")

            items_in_section = 0

            try:
                with self.db_manager.get_session() as session:
                    for media_data in self.plex_client.scan_library_section(section.title):
                        try:
                            item, created = self.db_manager.get_or_create_media_item(
                                session, **media_data
                            )

                            if created:
                                scan_stats["items_added"] += 1
                            else:
                                scan_stats["items_updated"] += 1

                            scan_stats["items_scanned"] += 1
                            items_in_section += 1

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
            if category_names:
                categories = [self.db_manager.get_category_by_name(session, name)
                             for name in category_names]
                categories = [c for c in categories if c is not None]
            else:
                categories = self.db_manager.get_collection_categories(session, enabled_only=True)

            if not categories:
                logger.warning("No categories found for clearing classifications")
                return

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

            all_unclassified_items = set()
            for category in categories:
                unclassified = self.db_manager.get_unclassified_items(session, category.id)
                all_unclassified_items.update(unclassified)

            all_unclassified_items = list(all_unclassified_items)

            if not all_unclassified_items:
                logger.info("No unclassified items found")
                return

            logger.info(f"Found {len(all_unclassified_items)} items needing classification across all categories")

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

            categories_data = []
            for category in categories:
                categories_data.append({
                    'id': category.id,
                    'name': category.name,
                    'description': category.description,
                    'prompt': category.prompt
                })

            total_progress = 0.5
            current_progress = 0.4

            if progress_callback:
                progress_callback(current_progress, "Starting mega-batch classification...")

            try:
                batch_results = await self.openai_client.classify_media_items(
                    media_data,
                    categories_data,
                    progress_callback=lambda p, msg: progress_callback(
                        current_progress + total_progress * p,
                        msg
                    ) if progress_callback else None
                )

                for batch_result in batch_results:
                    scan_stats["ai_requests_made"] += 1
                    scan_stats["total_tokens_used"] += batch_result.total_tokens
                    scan_stats["total_cost_estimate"] += batch_result.total_cost

                    self.db_manager.log_ai_processing(
                        session,
                        model_used=self.openai_client.model,
                        prompt_tokens=batch_result.total_tokens // 2,
                        completion_tokens=batch_result.total_tokens // 2,
                        items_in_batch=len(batch_result.results),
                        processing_time=batch_result.processing_time,
                        success=batch_result.success,
                        scan_id=self.current_scan_id,
                        error_message=batch_result.error_message,
                        estimated_cost=batch_result.total_cost
                    )

                    for result in batch_result.results:
                        try:
                            classification = self.db_manager.create_classification(
                                session,
                                media_item_id=result.media_item_id,
                                category_id=result.category_id,
                                matches=result.matches,
                                confidence=result.confidence,
                                ai_reasoning=result.reasoning,
                                model_used=self.openai_client.model,
                                tokens_used=result.tokens_used,
                                processing_time=result.processing_time
                            )
                            scan_stats["classifications_processed"] += 1

                        except Exception as e:
                            logger.error(f"Error storing classification: {e}")
                            scan_stats["items_failed"] += 1

                for category in categories:
                    self.db_manager.update_category_stats(session, category.id)

                logger.info(f"Mega-batch classification complete: {scan_stats['classifications_processed']} classifications processed")
                logger.info(f"Total cost: ${scan_stats['total_cost_estimate']:.4f}")

            except Exception as e:
                logger.error(f"Error in mega-batch classification: {e}")
                scan_stats["items_failed"] += len(all_unclassified_items) * len(categories)

        logger.info(f"Classification complete: {scan_stats['classifications_processed']} items classified")

    async def _update_plex_collections(self, scan_stats: Dict[str, Any],
                                     progress_callback: Optional[Callable]) -> None:
        """Update Plex collections based on classifications."""

        current_progress = 0.9

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
                    matching_items = self.db_manager.get_category_items(
                        session, category.id, matches_only=True
                    )

                    if not matching_items:
                        logger.info(f"No matching items for collection: {category.name}")
                        current_progress += progress_per_category
                        continue

                    # Group by library section
                    items_by_section: Dict[str, List[str]] = {}
                    for item in matching_items:
                        section = item.library_section
                        if section not in items_by_section:
                            items_by_section[section] = []
                        items_by_section[section].append(item.plex_key)

                    for section, item_keys in items_by_section.items():
                        logger.info(f"Updating collection '{category.name}' in {section} with {len(item_keys)} items")

                        try:
                            if self.plex_client.collection_exists(section, category.name):
                                success = self.plex_client.update_collection(
                                    section, category.name, item_keys
                                )
                                if success:
                                    scan_stats["collections_updated"] += 1
                                    logger.info(f"Updated collection '{category.name}' in {section}")
                            else:
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

    async def natural_language_collection(self, query: str,
                                         library_sections: Optional[List[str]] = None,
                                         auto_create: bool = False,
                                         progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Create a collection from a natural language query.

        Returns dict with: suggested_name, matching_items[], total_searched, query
        """
        if progress_callback:
            progress_callback(0.1, "Loading media items from database...")

        # Load all media items
        with self.db_manager.get_session() as session:
            if library_sections:
                all_items = []
                for section in library_sections:
                    items = self.db_manager.get_media_items(session, library_section=section)
                    all_items.extend(items)
            else:
                all_items = self.db_manager.get_media_items(session)

            if not all_items:
                return {
                    "suggested_name": "",
                    "matching_items": [],
                    "total_searched": 0,
                    "query": query,
                    "error": "No media items in database. Run a library scan first."
                }

            # Convert to dict format for AI client
            media_data = []
            item_lookup = {}
            for item in all_items:
                item_dict = {
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
                }
                media_data.append(item_dict)
                item_lookup[item.id] = item

            if progress_callback:
                progress_callback(0.2, f"Searching {len(media_data)} items with AI...")

            # Run NL search
            result = await self.openai_client.natural_language_search(
                media_data, query,
                progress_callback=lambda p, msg: progress_callback(
                    0.2 + 0.6 * p, msg
                ) if progress_callback else None
            )

            # Build matching items list
            matching_items = []
            for classification in result.results:
                if classification.matches:
                    item = item_lookup.get(classification.media_item_id)
                    if item:
                        matching_items.append({
                            'id': item.id,
                            'title': item.title,
                            'year': item.year,
                            'summary': (item.summary or '')[:200],
                            'genres': item.genres or [],
                            'plex_key': item.plex_key,
                            'library_section': item.library_section
                        })

            response = {
                "suggested_name": result.suggested_name or "Custom Collection",
                "matching_items": matching_items,
                "total_searched": len(media_data),
                "query": query,
                "total_tokens": result.total_tokens,
                "total_cost": result.total_cost
            }

            # Auto-create collection if requested
            if auto_create and matching_items:
                if progress_callback:
                    progress_callback(0.9, "Creating collection in Plex...")

                collection_name = result.suggested_name or "Custom Collection"

                # Save NL category to DB
                nl_category = self.db_manager.create_natural_language_collection(
                    session, collection_name, query
                )

                # Create classifications for matched items
                for item_data in matching_items:
                    self.db_manager.create_classification(
                        session,
                        media_item_id=item_data['id'],
                        category_id=nl_category.id,
                        matches=True,
                        confidence=0.8,
                        ai_reasoning=f"Matched natural language query: {query}",
                        model_used=self.openai_client.model
                    )

                # Create Plex collection
                items_by_section: Dict[str, List[str]] = {}
                for item_data in matching_items:
                    section = item_data['library_section']
                    if section not in items_by_section:
                        items_by_section[section] = []
                    items_by_section[section].append(item_data['plex_key'])

                for section, item_keys in items_by_section.items():
                    try:
                        self.plex_client.create_collection(
                            section, collection_name, item_keys,
                            f"AI-generated: {query}"
                        )
                        logger.info(f"Created NL collection '{collection_name}' in {section}")
                    except Exception as e:
                        logger.error(f"Failed to create NL collection in Plex: {e}")
                        response["error"] = f"Collection saved but Plex creation failed: {e}"

                response["collection_created"] = True

            if progress_callback:
                progress_callback(1.0, f"Found {len(matching_items)} matching movies!")

            return response

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
