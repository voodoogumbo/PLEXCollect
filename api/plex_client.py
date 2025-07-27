"""Plex server integration for PLEXCollect."""

import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Generator, Tuple
from plexapi.server import PlexServer
from plexapi.library import LibrarySection, MovieSection, ShowSection
from plexapi.video import Movie, Show, Episode
from plexapi.collection import Collection
from plexapi.exceptions import PlexApiException, NotFound, Unauthorized
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from utils.config import get_config
from models.database_models import MediaItem
from api.database import get_database_manager

logger = logging.getLogger(__name__)

class PlexClient:
    """Client for interacting with Plex server."""
    
    def __init__(self, server_url: Optional[str] = None, token: Optional[str] = None):
        """Initialize Plex client."""
        if server_url is None or token is None:
            config = get_config()
            server_url = server_url or config.plex.server_url
            token = token or config.plex.token
        
        self.server_url = server_url
        self.token = token
        self.plex: Optional[PlexServer] = None
        self._setup_session()
    
    def _setup_session(self) -> None:
        """Setup HTTP session with retry strategy."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set session timeout
        session.timeout = 30
        
        # Store session for plexapi to use
        self._session = session
    
    def connect(self) -> bool:
        """Connect to Plex server."""
        try:
            self.plex = PlexServer(
                self.server_url, 
                self.token,
                session=self._session
            )
            
            # Test connection by getting server info
            server_info = self.plex.identity
            logger.info(f"Connected to Plex server: {server_info}")
            return True
            
        except Unauthorized:
            logger.error("Plex authentication failed - check your token")
            return False
        except requests.ConnectionError:
            logger.error(f"Cannot connect to Plex server at {self.server_url}")
            return False
        except PlexApiException as e:
            logger.error(f"Plex API error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Plex: {e}")
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection and return server info."""
        try:
            if not self.plex:
                if not self.connect():
                    return {
                        "success": False,
                        "error": "Failed to connect to Plex server"
                    }
            
            server_info = {
                "success": True,
                "server_name": self.plex.friendlyName,
                "server_version": self.plex.version,
                "platform": self.plex.platform,
                "platform_version": self.plex.platformVersion,
                "library_sections": []
            }
            
            # Get library sections info
            for section in self.plex.library.sections():
                section_info = {
                    "title": section.title,
                    "type": section.type,
                    "key": section.key,
                    "item_count": len(section.all()) if hasattr(section, 'all') else 0
                }
                server_info["library_sections"].append(section_info)
            
            return server_info
            
        except Exception as e:
            logger.error(f"Error testing Plex connection: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_library_sections(self, section_types: Optional[List[str]] = None) -> List[LibrarySection]:
        """Get library sections, optionally filtered by type."""
        if not self.plex:
            raise RuntimeError("Not connected to Plex server")
        
        sections = self.plex.library.sections()
        
        if section_types:
            sections = [s for s in sections if s.type in section_types]
        
        return sections
    
    def scan_library_section(self, section_name: str, 
                           incremental: bool = False) -> Generator[Dict[str, Any], None, None]:
        """Scan a library section and yield media items."""
        if not self.plex:
            raise RuntimeError("Not connected to Plex server")
        
        try:
            section = self.plex.library.section(section_name)
            logger.info(f"Scanning library section: {section_name} ({section.type})")
            
            # Get all items from section
            if hasattr(section, 'all'):
                items = section.all()
                logger.info(f"Found {len(items)} items in {section_name}")
                
                for item in items:
                    try:
                        media_data = self._extract_media_data(item, section_name)
                        if media_data:
                            yield media_data
                    except Exception as e:
                        logger.warning(f"Error processing item {getattr(item, 'title', 'Unknown')}: {e}")
                        continue
            else:
                logger.warning(f"Library section {section_name} does not support scanning")
                
        except NotFound:
            logger.error(f"Library section '{section_name}' not found")
            raise
        except Exception as e:
            logger.error(f"Error scanning library section {section_name}: {e}")
            raise
    
    def _extract_media_data(self, item: Any, library_section: str) -> Optional[Dict[str, Any]]:
        """Extract metadata from a Plex media item."""
        try:
            # Common fields for all media types
            data = {
                "plex_key": str(item.key),
                "title": item.title,
                "library_section": library_section,
                "summary": getattr(item, 'summary', None),
                "added_at": getattr(item, 'addedAt', None),
                "updated_at": getattr(item, 'updatedAt', None),
            }
            
            # Type-specific fields
            if isinstance(item, Movie):
                data.update(self._extract_movie_data(item))
            elif isinstance(item, Show):
                data.update(self._extract_show_data(item))
            elif isinstance(item, Episode):
                data.update(self._extract_episode_data(item))
            else:
                logger.debug(f"Unsupported item type: {type(item)}")
                return None
            
            return data
            
        except Exception as e:
            logger.error(f"Error extracting data from {getattr(item, 'title', 'Unknown')}: {e}")
            return None
    
    def _extract_movie_data(self, movie: Movie) -> Dict[str, Any]:
        """Extract movie-specific metadata."""
        return {
            "type": "movie",
            "year": getattr(movie, 'year', None),
            "genres": [genre.tag for genre in getattr(movie, 'genres', [])],
            "directors": [director.tag for director in getattr(movie, 'directors', [])],
            "actors": [actor.tag for actor in getattr(movie, 'actors', [])[:10]],  # Limit actors
            "studios": [studio.tag for studio in getattr(movie, 'studios', [])],
            "collections": [collection.tag for collection in getattr(movie, 'collections', [])],
            "content_rating": getattr(movie, 'contentRating', None),
            "duration": getattr(movie, 'duration', None),  # in milliseconds
            "rating": getattr(movie, 'rating', None),
        }
    
    def _extract_show_data(self, show: Show) -> Dict[str, Any]:
        """Extract TV show-specific metadata."""
        return {
            "type": "show",
            "year": getattr(show, 'year', None),
            "genres": [genre.tag for genre in getattr(show, 'genres', [])],
            "directors": [],  # Shows typically don't have directors at show level
            "actors": [actor.tag for actor in getattr(show, 'actors', [])[:10]],
            "studios": [studio.tag for studio in getattr(show, 'studios', [])],
            "collections": [collection.tag for collection in getattr(show, 'collections', [])],
            "content_rating": getattr(show, 'contentRating', None),
            "duration": None,  # Shows don't have duration
            "rating": getattr(show, 'rating', None),
        }
    
    def _extract_episode_data(self, episode: Episode) -> Dict[str, Any]:
        """Extract episode-specific metadata."""
        show = episode.show()
        return {
            "type": "episode",
            "year": getattr(show, 'year', None),
            "genres": [genre.tag for genre in getattr(show, 'genres', [])],
            "directors": [director.tag for director in getattr(episode, 'directors', [])],
            "actors": [actor.tag for actor in getattr(episode, 'actors', [])[:10]],
            "studios": [studio.tag for studio in getattr(show, 'studios', [])],
            "collections": [],  # Episodes typically don't have collections
            "content_rating": getattr(episode, 'contentRating', None),
            "duration": getattr(episode, 'duration', None),
            "rating": getattr(episode, 'rating', None),
        }
    
    def get_collections(self, library_section: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get existing collections from Plex."""
        if not self.plex:
            raise RuntimeError("Not connected to Plex server")
        
        collections = []
        sections = [self.plex.library.section(library_section)] if library_section else self.plex.library.sections()
        
        for section in sections:
            try:
                if hasattr(section, 'collections'):
                    for collection in section.collections():
                        collections.append({
                            "title": collection.title,
                            "key": collection.key,
                            "summary": getattr(collection, 'summary', ''),
                            "item_count": len(collection.items()),
                            "library_section": section.title,
                            "created_at": getattr(collection, 'addedAt', None),
                            "updated_at": getattr(collection, 'updatedAt', None)
                        })
            except Exception as e:
                logger.warning(f"Error getting collections from section {section.title}: {e}")
        
        return collections
    
    def create_collection(self, library_section: str, name: str, 
                         items: List[str], summary: Optional[str] = None) -> bool:
        """Create a new collection in Plex."""
        if not self.plex:
            raise RuntimeError("Not connected to Plex server")
        
        try:
            section = self.plex.library.section(library_section)
            
            # Get media items by their keys
            media_items = []
            for item_key in items:
                try:
                    item = self.plex.fetchItem(item_key)
                    media_items.append(item)
                except Exception as e:
                    logger.warning(f"Could not fetch item {item_key}: {e}")
            
            if not media_items:
                logger.warning(f"No valid items found for collection {name}")
                return False
            
            # Create collection
            collection = section.createCollection(
                title=name,
                items=media_items
            )
            
            # Set summary if provided
            if summary and hasattr(collection, 'editSummary'):
                collection.editSummary(summary)
            
            logger.info(f"Created collection '{name}' with {len(media_items)} items")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            return False
    
    def update_collection(self, library_section: str, collection_name: str, 
                         items_to_add: List[str], items_to_remove: List[str] = None) -> bool:
        """Update an existing collection."""
        if not self.plex:
            raise RuntimeError("Not connected to Plex server")
        
        try:
            section = self.plex.library.section(library_section)
            
            # Find the collection
            collection = None
            for coll in section.collections():
                if coll.title == collection_name:
                    collection = coll
                    break
            
            if not collection:
                logger.error(f"Collection '{collection_name}' not found in {library_section}")
                return False
            
            # Add items
            if items_to_add:
                media_items_to_add = []
                for item_key in items_to_add:
                    try:
                        item = self.plex.fetchItem(item_key)
                        media_items_to_add.append(item)
                    except Exception as e:
                        logger.warning(f"Could not fetch item to add {item_key}: {e}")
                
                if media_items_to_add:
                    collection.addItems(media_items_to_add)
                    logger.info(f"Added {len(media_items_to_add)} items to collection '{collection_name}'")
            
            # Remove items
            if items_to_remove:
                media_items_to_remove = []
                for item_key in items_to_remove:
                    try:
                        item = self.plex.fetchItem(item_key)
                        media_items_to_remove.append(item)
                    except Exception as e:
                        logger.warning(f"Could not fetch item to remove {item_key}: {e}")
                
                if media_items_to_remove:
                    collection.removeItems(media_items_to_remove)
                    logger.info(f"Removed {len(media_items_to_remove)} items from collection '{collection_name}'")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating collection {collection_name}: {e}")
            return False
    
    def collection_exists(self, library_section: str, collection_name: str) -> bool:
        """Check if a collection exists."""
        if not self.plex:
            return False
        
        try:
            section = self.plex.library.section(library_section)
            for collection in section.collections():
                if collection.title == collection_name:
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking if collection exists: {e}")
            return False

# Global Plex client instance
_plex_client: Optional[PlexClient] = None

def get_plex_client() -> PlexClient:
    """Get the global Plex client instance."""
    global _plex_client
    if _plex_client is None:
        _plex_client = PlexClient()
    return _plex_client