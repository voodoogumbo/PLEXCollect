"""OpenAI integration for PLEXCollect with batch processing and rate limiting."""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI
from openai.types.chat import ChatCompletion

from utils.config import get_config
from models.database_models import MediaItem, CollectionCategory
from api.database import get_database_manager

logger = logging.getLogger(__name__)

@dataclass
class ClassificationResult:
    """Result of AI classification for a media item."""
    media_item_id: int
    category_id: int
    matches: bool
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    tokens_used: int = 0
    processing_time: float = 0.0
    error: Optional[str] = None
    
    # Franchise-specific fields
    franchise_position: Optional[int] = None
    franchise_year: Optional[int] = None
    franchise_reasoning: Optional[str] = None

@dataclass
class BatchResult:
    """Result of batch processing."""
    results: List[ClassificationResult]
    total_tokens: int
    total_cost: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class RateLimiter:
    """Rate limiter for OpenAI API calls."""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        
        # Tracking
        self.request_times: List[datetime] = []
        self.token_usage: List[Tuple[datetime, int]] = []
        
        # Lock for thread safety (created lazily)
        self._lock = None
    
    async def acquire(self, estimated_tokens: int = 1000) -> None:
        """Acquire rate limit permission before making a request."""
        # Create lock on first use
        if self._lock is None:
            self._lock = asyncio.Lock()
        
        async with self._lock:
            now = datetime.now()
            one_minute_ago = now - timedelta(minutes=1)
            
            # Clean old entries
            self.request_times = [t for t in self.request_times if t > one_minute_ago]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > one_minute_ago]
            
            # Check request rate limit
            if len(self.request_times) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.request_times[0]).total_seconds()
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire(estimated_tokens)
            
            # Check token rate limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.tokens_per_minute:
                sleep_time = 60 - (now - self.token_usage[0][0]).total_seconds()
                if sleep_time > 0:
                    logger.info(f"Token rate limit reached, sleeping for {sleep_time:.1f} seconds")
                    await asyncio.sleep(sleep_time)
                    return await self.acquire(estimated_tokens)
            
            # Record this request
            self.request_times.append(now)
            self.token_usage.append((now, estimated_tokens))

class OpenAIClient:
    """Client for OpenAI API with batch processing and rate limiting."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize OpenAI client."""
        config = get_config()
        
        self.api_key = api_key or config.openai.api_key
        self.model = model or config.openai.model
        self.max_tokens = config.openai.max_tokens
        self.temperature = config.openai.temperature
        self.batch_size = config.openai.batch_size
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            config.openai.rate_limit.requests_per_minute,
            config.openai.rate_limit.tokens_per_minute
        )
        
        # Cost tracking (approximate)
        self.cost_per_1k_tokens = self._get_cost_per_1k_tokens()
    
    def _get_cost_per_1k_tokens(self) -> float:
        """Get approximate cost per 1000 tokens for the model."""
        # These are approximate costs and may change
        # Order matters - more specific models first
        costs = [
            ("o4-mini", 0.001),  # o4-mini pricing (estimated)
            ("o1-mini", 0.003),  # o1-mini pricing
            ("o1-preview", 0.015),  # o1-preview pricing
            ("gpt-4o-mini", 0.0001),
            ("gpt-4o", 0.005),
            ("gpt-4-turbo", 0.01),
            ("gpt-4", 0.03),
            ("gpt-3.5-turbo", 0.002),
        ]
        
        for model_prefix, cost in costs:
            if self.model.startswith(model_prefix):
                return cost
        
        return 0.01  # Default estimate
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of tokens in text."""
        # Very rough estimation: ~4 characters per token
        return len(text) // 4 + 100  # Add buffer for system message
    
    def _build_classification_prompt(self, media_items: List[Dict[str, Any]], 
                                   category: Dict[str, Any]) -> str:
        """Build the classification prompt for a batch of media items."""
        
        is_franchise = category.get('is_franchise', False)
        
        if is_franchise:
            return self._build_franchise_prompt(media_items, category)
        else:
            return self._build_regular_prompt(media_items, category)
    
    def _build_regular_prompt(self, media_items: List[Dict[str, Any]], 
                            category: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompt for regular (non-franchise) collections."""
        
        system_prompt = f"""You are a movie and TV show classifier. Your task is to determine which items belong to the category "{category['name']}".

Category Description: {category['description']}
Classification Rule: {category['prompt']}

For each media item, respond with JSON containing:
- "matches": true/false if the item belongs to this category
- "confidence": 0.0-1.0 confidence score
- "reasoning": Brief explanation for your decision

Consider the title, summary, genres, year, and other metadata when making your decision.
Be consistent and accurate in your classifications."""

        items_text = "Items to classify:\n"
        for i, item in enumerate(media_items, 1):
            item_text = f"""
{i}. Title: {item.get('title', 'Unknown')}
   Year: {item.get('year', 'Unknown')}
   Type: {item.get('type', 'Unknown')}
   Genres: {', '.join(item.get('genres', []))}
   Summary: {item.get('summary', 'No summary available')[:500]}...
   """
            items_text += item_text
        
        user_prompt = f"""{items_text}

Respond with a JSON object containing a "results" array where each object corresponds to the item at that index:
{{
  "results": [
    {{"matches": true, "confidence": 0.9, "reasoning": "explanation"}},
    {{"matches": false, "confidence": 0.1, "reasoning": "explanation"}},
    ...
  ]
}}"""
        
        return system_prompt, user_prompt
    
    def _build_franchise_prompt(self, media_items: List[Dict[str, Any]], 
                              category: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompt for franchise collections with chronological ordering."""
        
        franchise_name = category['name']
        
        # Create franchise-specific knowledge based on the category
        franchise_info = self._get_franchise_chronology_info(franchise_name)
        
        system_prompt = f"""You are a movie franchise expert specializing in chronological ordering. Your task is to determine which movies belong to the "{franchise_name}" and their chronological position in the story timeline.

Franchise: {franchise_name}
Description: {category['description']}

{franchise_info}

For each media item, respond with JSON containing:
- "matches": true/false if the item belongs to this franchise
- "confidence": 0.0-1.0 confidence score for franchise membership
- "chronological_position": integer position in story timeline (1, 2, 3, etc.) if matches=true, null if matches=false
- "franchise_year": approximate in-universe year or era if known, null otherwise
- "reasoning": Brief explanation for your franchise classification and chronological positioning

IMPORTANT: Use CHRONOLOGICAL (story) order, NOT release order. For example:
- Star Wars Episode I comes before Episode IV chronologically
- X-Men: First Class comes before the original X-Men movie chronologically
- Fast & Furious: Tokyo Drift comes later in the timeline than its release suggests

Consider the title, plot, characters, timeline references, and story connections."""

        items_text = "Items to classify:\n"
        for i, item in enumerate(media_items, 1):
            item_text = f"""
{i}. Title: {item.get('title', 'Unknown')}
   Year: {item.get('year', 'Unknown')}
   Type: {item.get('type', 'Unknown')}
   Genres: {', '.join(item.get('genres', []))}
   Directors: {', '.join(item.get('directors', []))}
   Summary: {item.get('summary', 'No summary available')[:500]}...
   """
            items_text += item_text
        
        user_prompt = f"""{items_text}

Respond with a JSON object containing a "results" array where each object corresponds to the item at that index:
{{
  "results": [
    {{"matches": true, "confidence": 0.9, "chronological_position": 1, "franchise_year": 1977, "reasoning": "explanation"}},
    {{"matches": false, "confidence": 0.1, "chronological_position": null, "franchise_year": null, "reasoning": "explanation"}},
    ...
  ]
}}"""
        
        return system_prompt, user_prompt
    
    def _get_franchise_chronology_info(self, franchise_name: str) -> str:
        """Get franchise-specific chronological information to help AI ordering."""
        
        franchise_guides = {
            "Star Wars Saga": """
Chronological Order (Story Timeline):
1. Episode I: The Phantom Menace (1999) - 32 BBY
2. Episode II: Attack of the Clones (2002) - 22 BBY  
3. Episode III: Revenge of the Sith (2005) - 19 BBY
4. Solo: A Star Wars Story (2018) - 13-10 BBY
5. Rogue One: A Star Wars Story (2016) - 0 BBY
6. Episode IV: A New Hope (1977) - 0 ABY
7. Episode V: The Empire Strikes Back (1980) - 3 ABY
8. Episode VI: Return of the Jedi (1983) - 4 ABY
9. Episode VII: The Force Awakens (2015) - 34 ABY
10. Episode VIII: The Last Jedi (2017) - 34 ABY
11. Episode IX: The Rise of Skywalker (2019) - 35 ABY""",
            
            "Marvel Cinematic Universe": """
Chronological Order (MCU Timeline):
1. Captain America: The First Avenger (2011) - 1940s
2. Captain Marvel (2019) - 1995
3. Iron Man (2008) - 2010
4. Iron Man 2 (2010) - 2011
5. The Incredible Hulk (2008) - 2011
6. Thor (2011) - 2011
7. The Avengers (2012) - 2012
8. Iron Man 3 (2013) - 2012
9. Thor: The Dark World (2013) - 2013
10. Captain America: The Winter Soldier (2014) - 2014
11. Guardians of the Galaxy (2014) - 2014
12. Guardians of the Galaxy Vol. 2 (2017) - 2014
13. Avengers: Age of Ultron (2015) - 2015
14. Ant-Man (2015) - 2015
15. Captain America: Civil War (2016) - 2016
16. Black Widow (2021) - 2016
17. Spider-Man: Homecoming (2017) - 2016
18. Doctor Strange (2016) - 2016-2017
19. Black Panther (2018) - 2016
20. Thor: Ragnarok (2017) - 2017
21. Ant-Man and the Wasp (2018) - 2018
22. Avengers: Infinity War (2018) - 2018
23. Avengers: Endgame (2019) - 2018-2023
24. Spider-Man: Far From Home (2019) - 2024""",
            
            "Fast & Furious Franchise": """
Chronological Order (Story Timeline):
1. The Fast and the Furious (2001)
2. 2 Fast 2 Furious (2003)
3. Fast & Furious (2009)
4. Fast Five (2011)
5. Fast & Furious 6 (2013)
6. The Fast and the Furious: Tokyo Drift (2006) - Takes place after Fast 6
7. Furious 7 (2015)
8. The Fate of the Furious (2017)
9. Fast & Furious Presents: Hobbs & Shaw (2019) - Spin-off
10. F9: The Fast Saga (2021)""",
            
            "Indiana Jones Adventures": """
Chronological Order (Story Timeline):
1. Indiana Jones and the Temple of Doom (1984) - 1935
2. Raiders of the Lost Ark (1981) - 1936
3. Indiana Jones and the Last Crusade (1989) - 1938
4. Indiana Jones and the Kingdom of the Crystal Skull (2008) - 1957
5. Indiana Jones and the Dial of Destiny (2023) - 1969""",
            
            "X-Men Universe": """
Chronological Order (Story Timeline):
1. X-Men: First Class (2011) - 1962
2. X-Men: Days of Future Past (2014) - 1973 (past timeline)
3. X-Men Origins: Wolverine (2009) - 1979
4. X-Men: Apocalypse (2016) - 1983
5. X-Men: Dark Phoenix (2019) - 1992
6. X-Men (2000) - 2003
7. X2: X-Men United (2003) - 2003
8. X-Men: The Last Stand (2006) - 2006
9. The Wolverine (2013) - 2013
10. Deadpool (2016) - 2016
11. Logan (2017) - 2029"""
        }
        
        return franchise_guides.get(franchise_name, 
            f"Franchise: {franchise_name}\nUse your knowledge of this franchise to determine chronological story order.")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError))
    )
    async def _make_api_call(self, system_prompt: str, user_prompt: str) -> ChatCompletion:
        """Make API call with retry logic."""
        
        estimated_tokens = self._estimate_tokens(system_prompt + user_prompt)
        await self.rate_limiter.acquire(estimated_tokens)
        
        start_time = time.time()
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Handle different model parameter requirements
            if self.model.startswith("o1"):
                # o1 models don't support max_tokens, temperature, or response_format
                pass
            elif self.model.startswith("o4"):
                # o4 models use max_completion_tokens instead of max_tokens
                api_params["max_completion_tokens"] = self.max_tokens
                # o4 models don't support temperature or response_format
            else:
                # Standard models support all parameters
                api_params["max_tokens"] = self.max_tokens
                api_params["temperature"] = self.temperature
                api_params["response_format"] = {"type": "json_object"}
            
            response = self.client.chat.completions.create(**api_params)
            
            processing_time = time.time() - start_time
            logger.debug(f"OpenAI API call completed in {processing_time:.2f}s with model {self.model}")
            
            # Check for truncated response (not applicable to o1/o4 models)
            if not (self.model.startswith("o1") or self.model.startswith("o4")):
                if hasattr(response.choices[0], 'finish_reason') and response.choices[0].finish_reason == 'length':
                    logger.warning("Response was truncated due to token limit")
                    raise ValueError("Response truncated - chunk too large")
            
            return response
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit error: {e}")
            raise
        except openai.APITimeoutError as e:
            logger.warning(f"API timeout error: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def _parse_classification_response(self, response: ChatCompletion, 
                                     media_items: List[Dict[str, Any]],
                                     category_id: int,
                                     is_franchise: bool = False) -> List[ClassificationResult]:
        """Parse the OpenAI response into classification results."""
        results = []
        
        try:
            # Extract the response content
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")
            
            # Debug logging for API responses
            logger.debug(f"OpenAI response content: {content[:500]}...")  # First 500 chars
            
            # Parse JSON response
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's wrapped in text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            # Extract the results array from the response object
            if isinstance(response_data, dict) and "results" in response_data:
                classifications = response_data["results"]
            elif isinstance(response_data, list):
                # Fallback for old format
                classifications = response_data
            else:
                raise ValueError("Response does not contain 'results' array")
            
            if not isinstance(classifications, list):
                raise ValueError("Classifications is not a list")
            
            # Validate we have the right number of results
            if len(classifications) != len(media_items):
                logger.warning(f"Mismatch: {len(classifications)} results for {len(media_items)} items")
            
            # Process each classification
            for i, (item, classification) in enumerate(zip(media_items, classifications)):
                if not isinstance(classification, dict):
                    logger.warning(f"Invalid classification format for item {i}")
                    results.append(ClassificationResult(
                        media_item_id=item['id'],
                        category_id=category_id,
                        matches=False,
                        error="Invalid response format"
                    ))
                    continue
                
                # Create classification result with franchise fields if applicable
                result = ClassificationResult(
                    media_item_id=item['id'],
                    category_id=category_id,
                    matches=classification.get('matches', False),
                    confidence=classification.get('confidence'),
                    reasoning=classification.get('reasoning'),
                    tokens_used=response.usage.total_tokens // len(media_items),  # Approximate
                    processing_time=0.0  # Will be set by caller
                )
                
                # Add franchise-specific data if this is a franchise classification
                if is_franchise:
                    result.franchise_position = classification.get('chronological_position')
                    result.franchise_year = classification.get('franchise_year')
                    result.franchise_reasoning = classification.get('reasoning')  # Use same reasoning for now
                
                results.append(result)
            
        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
            # Return error results for all items
            for item in media_items:
                results.append(ClassificationResult(
                    media_item_id=item['id'],
                    category_id=category_id,
                    matches=False,
                    error=f"Parse error: {str(e)}"
                ))
        
        return results
    
    async def classify_media_batch(self, media_items: List[Dict[str, Any]], 
                                 category: Dict[str, Any]) -> BatchResult:
        """Classify a batch of media items for a specific category."""
        start_time = time.time()
        
        try:
            # Limit batch size
            if len(media_items) > self.batch_size:
                media_items = media_items[:self.batch_size]
                logger.warning(f"Batch size limited to {self.batch_size} items")
            
            logger.info(f"Classifying {len(media_items)} items for category '{category['name']}'")
            
            # Build prompt
            system_prompt, user_prompt = self._build_classification_prompt(media_items, category)
            
            # Make API call
            response = await self._make_api_call(system_prompt, user_prompt)
            
            # Parse results
            is_franchise = category.get('is_franchise', False)
            results = self._parse_classification_response(response, media_items, category['id'], is_franchise)
            
            # Calculate metrics
            total_tokens = response.usage.total_tokens
            total_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            processing_time = time.time() - start_time
            
            # Update processing time for each result
            for result in results:
                result.processing_time = processing_time / len(results)
            
            # Log statistics
            successful_classifications = sum(1 for r in results if not r.error)
            matches = sum(1 for r in results if r.matches and not r.error)
            
            logger.info(f"Batch completed: {successful_classifications}/{len(results)} successful, "
                       f"{matches} matches, {total_tokens} tokens, ${total_cost:.4f}")
            
            return BatchResult(
                results=results,
                total_tokens=total_tokens,
                total_cost=total_cost,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Batch classification failed: {e}")
            
            # Return error results
            error_results = [
                ClassificationResult(
                    media_item_id=item['id'],
                    category_id=category['id'],
                    matches=False,
                    error=str(e)
                ) for item in media_items
            ]
            
            return BatchResult(
                results=error_results,
                total_tokens=0,
                total_cost=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def classify_media_items(self, media_items: List[Dict[str, Any]], 
                                 categories: List[Dict[str, Any]],
                                 progress_callback: Optional[callable] = None) -> List[BatchResult]:
        """Classify all media items across all categories in a single optimized API call."""
        logger.info(f"Starting mega-batch classification of {len(media_items)} items across {len(categories)} categories")
        
        if progress_callback:
            progress_callback(0.1, "Building comprehensive classification prompt...")
        
        try:
            # Use the new mega-batch approach for maximum efficiency
            result = await self.classify_all_items_mega_batch(media_items, categories, progress_callback)
            
            logger.info(f"Mega-batch classification complete: {result.total_tokens} tokens, ${result.total_cost:.4f} cost")
            return [result]
            
        except Exception as e:
            logger.error(f"Mega-batch classification failed, falling back to old method: {e}")
            
            # Fallback to old batching method if mega-batch fails
            return await self.classify_media_items_legacy(media_items, categories, progress_callback)
    
    async def classify_media_items_legacy(self, media_items: List[Dict[str, Any]], 
                                        categories: List[Dict[str, Any]],
                                        progress_callback: Optional[callable] = None) -> List[BatchResult]:
        """Legacy batching method - fallback only."""
        all_results = []
        total_batches = len(categories) * ((len(media_items) + self.batch_size - 1) // self.batch_size)
        completed_batches = 0
        
        logger.info(f"Using legacy batching for {len(media_items)} items across {len(categories)} categories")
        
        for category in categories:
            logger.info(f"Processing category: {category['name']}")
            
            # Process items in batches
            for i in range(0, len(media_items), self.batch_size):
                batch = media_items[i:i + self.batch_size]
                
                try:
                    result = await self.classify_media_batch(batch, category)
                    all_results.append(result)
                    
                    completed_batches += 1
                    
                    if progress_callback:
                        progress = completed_batches / total_batches
                        progress_callback(progress, f"Processed category '{category['name']}' batch {i//self.batch_size + 1}")
                    
                    # Small delay between batches to be nice to the API
                    if i + self.batch_size < len(media_items):
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Failed to process batch for category {category['name']}: {e}")
                    # Continue with next batch
                    continue
        
        # Summary statistics
        total_tokens = sum(r.total_tokens for r in all_results)
        total_cost = sum(r.total_cost for r in all_results)
        successful_batches = sum(1 for r in all_results if r.success)
        
        logger.info(f"Legacy classification complete: {successful_batches}/{len(all_results)} batches successful, "
                   f"{total_tokens} total tokens, ${total_cost:.4f} total cost")
        
        return all_results
    
    async def classify_all_items_mega_batch(self, media_items: List[Dict[str, Any]], 
                                          categories: List[Dict[str, Any]],
                                          progress_callback: Optional[callable] = None) -> BatchResult:
        """Mega-batch classification: Process all items across all categories in optimized chunks."""
        start_time = time.time()
        
        # Limit mega-batch size to prevent token overflow and JSON truncation
        # Reduced from 100 to 40 to ensure responses stay well under token limits
        MAX_MOVIES_PER_MEGA_BATCH = 40
        
        if len(media_items) > MAX_MOVIES_PER_MEGA_BATCH:
            logger.info(f"Large library detected ({len(media_items)} items). Using chunked mega-batch approach.")
            return await self._classify_chunked_mega_batch(media_items, categories, progress_callback)
        
        if progress_callback:
            progress_callback(0.2, "Building JSON mega-batch payload...")
        
        try:
            # Build the comprehensive system and user prompts
            system_prompt, user_prompt = self._build_mega_batch_prompt(media_items, categories)
            
            if progress_callback:
                progress_callback(0.4, "Sending JSON mega-batch to OpenAI...")
            
            # Make the single API call with retry on truncation
            try:
                response = await self._make_api_call(system_prompt, user_prompt)
            except ValueError as e:
                if "truncated" in str(e).lower() and len(media_items) > 20:
                    logger.warning(f"Response truncated with {len(media_items)} items, retrying with chunked approach")
                    return await self._classify_chunked_mega_batch(media_items, categories, progress_callback)
                else:
                    raise
            
            if progress_callback:
                progress_callback(0.7, "Parsing JSON classification results...")
            
            # Parse the mega-batch response
            all_results = self._parse_mega_batch_response(response, media_items, categories)
            
            if progress_callback:
                progress_callback(0.9, "Finalizing JSON mega-batch results...")
            
            # Calculate metrics
            total_tokens = response.usage.total_tokens
            total_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            processing_time = time.time() - start_time
            
            # Update processing time for each result
            for result in all_results:
                result.processing_time = processing_time / len(all_results)
            
            # Log statistics
            successful_classifications = sum(1 for r in all_results if not r.error)
            matches = sum(1 for r in all_results if r.matches and not r.error)
            
            logger.info(f"Mega-batch completed: {successful_classifications}/{len(all_results)} successful, "
                       f"{matches} matches, {total_tokens} tokens, ${total_cost:.4f}")
            
            if progress_callback:
                progress_callback(1.0, "JSON mega-batch classification complete!")
            
            return BatchResult(
                results=all_results,
                total_tokens=total_tokens,
                total_cost=total_cost,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Mega-batch classification failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return error results
            error_results = []
            for item in media_items:
                for category in categories:
                    error_results.append(ClassificationResult(
                        media_item_id=item['id'],
                        category_id=category['id'],
                        matches=False,
                        error=str(e)
                    ))
            
            return BatchResult(
                results=error_results,
                total_tokens=0,
                total_cost=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    async def _classify_chunked_mega_batch(self, media_items: List[Dict[str, Any]], 
                                         categories: List[Dict[str, Any]],
                                         progress_callback: Optional[callable] = None) -> BatchResult:
        """Process large libraries in chunks while maintaining mega-batch benefits."""
        MAX_MOVIES_PER_CHUNK = 40
        start_time = time.time()
        
        all_results = []
        total_tokens = 0
        total_cost = 0.0
        
        # Split into chunks
        chunks = [media_items[i:i + MAX_MOVIES_PER_CHUNK] 
                 for i in range(0, len(media_items), MAX_MOVIES_PER_CHUNK)]
        
        logger.info(f"Processing {len(media_items)} items in {len(chunks)} mega-batch chunks")
        
        for chunk_idx, chunk in enumerate(chunks):
            if progress_callback:
                chunk_progress = (chunk_idx / len(chunks)) * 0.9  # Reserve 10% for final processing
                progress_callback(chunk_progress, f"Processing mega-batch chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)")
            
            try:
                # Process this chunk as a mega-batch
                chunk_result = await self.classify_all_items_mega_batch(chunk, categories)
                
                all_results.extend(chunk_result.results)
                total_tokens += chunk_result.total_tokens
                total_cost += chunk_result.total_cost
                
                logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} completed: {len(chunk_result.results)} classifications")
                
            except ValueError as e:
                if "truncated" in str(e).lower() and len(chunk) > 10:
                    logger.warning(f"Chunk {chunk_idx + 1} truncated with {len(chunk)} items, splitting into smaller chunks")
                    # Split chunk in half and retry
                    mid = len(chunk) // 2
                    sub_chunks = [chunk[:mid], chunk[mid:]]
                    
                    for sub_chunk in sub_chunks:
                        try:
                            sub_result = await self.classify_all_items_mega_batch(sub_chunk, categories)
                            all_results.extend(sub_result.results)
                            total_tokens += sub_result.total_tokens
                            total_cost += sub_result.total_cost
                        except Exception as sub_e:
                            logger.error(f"Sub-chunk failed: {sub_e}")
                            # Add error results for failed sub-chunk
                            for item in sub_chunk:
                                for category in categories:
                                    all_results.append(ClassificationResult(
                                        media_item_id=item['id'],
                                        category_id=category['id'],
                                        matches=False,
                                        error=f"Sub-chunk processing failed: {str(sub_e)}"
                                    ))
                else:
                    logger.error(f"Chunk {chunk_idx + 1} failed: {e}")
                    # Add error results for this chunk
                    for item in chunk:
                        for category in categories:
                            all_results.append(ClassificationResult(
                                media_item_id=item['id'],
                                category_id=category['id'],
                                matches=False,
                                error=f"Chunk processing failed: {str(e)}"
                            ))
            except Exception as e:
                logger.error(f"Chunk {chunk_idx + 1} failed: {e}")
                # Add error results for this chunk
                for item in chunk:
                    for category in categories:
                        all_results.append(ClassificationResult(
                            media_item_id=item['id'],
                            category_id=category['id'],
                            matches=False,
                            error=f"Chunk processing failed: {str(e)}"
                        ))
        
        processing_time = time.time() - start_time
        
        # Update processing time for all results
        for result in all_results:
            result.processing_time = processing_time / len(all_results)
        
        # Log final statistics
        successful_classifications = sum(1 for r in all_results if not r.error)
        matches = sum(1 for r in all_results if r.matches and not r.error)
        
        logger.info(f"Chunked mega-batch completed: {successful_classifications}/{len(all_results)} successful, "
                   f"{matches} matches, {total_tokens} tokens, ${total_cost:.4f}")
        
        if progress_callback:
            progress_callback(1.0, f"Chunked mega-batch complete! {matches} franchise movies detected")
        
        return BatchResult(
            results=all_results,
            total_tokens=total_tokens,
            total_cost=total_cost,
            processing_time=processing_time,
            success=True
        )
    
    def _build_mega_batch_prompt(self, media_items: List[Dict[str, Any]], 
                                categories: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Build JSON-based mega-batch prompt with structured data payload."""
        
        # Create structured data payload for movies
        movies_payload = []
        for i, item in enumerate(media_items, 1):
            movie_data = {
                "id": i,
                "media_item_id": item['id'],
                "title": item.get('title', 'Unknown'),
                "year": item.get('year'),
                "type": item.get('type', 'movie'),
                "genres": item.get('genres', []),
                "directors": item.get('directors', []),
                "actors": item.get('actors', []),
                "summary": item.get('summary', '')[:400],  # Limit summary length
                "content_rating": item.get('content_rating'),
                "rating": item.get('rating')
            }
            movies_payload.append(movie_data)
        
        # Create structured data payload for categories
        categories_payload = []
        franchise_guides = {}
        
        for category in categories:
            is_franchise = category.get('is_franchise', False)
            
            category_data = {
                "id": category['id'],
                "name": category['name'],
                "description": category['description'],
                "classification_rule": category['prompt'],
                "is_franchise": is_franchise,
                "chronological_sorting": category.get('chronological_sorting', False)
            }
            
            # Add franchise guidance if needed
            if is_franchise:
                franchise_info = self._get_franchise_chronology_info(category['name'])
                if franchise_info:
                    category_data["franchise_timeline"] = franchise_info
                    franchise_guides[category['name']] = franchise_info
            
            categories_payload.append(category_data)
        
        # Build the system prompt with ultra-minimal index-based approach
        system_prompt = f"""You are Plex-Classifier v4, an ultra-efficient movie collection organizer. 

Your task: Classify movies into collections using minimal index arrays.

Classification Guidelines:
- Use confidence >= 0.6 to include a movie in a collection
- For franchise categories, arrange by chronological story order (NOT release order)
- Consider all metadata: title, year, genres, directors, summary
- Return movie indices only (numbers), not titles or details

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown, no code blocks.
Response format: JSON object with category names as keys and arrays of movie indices as values.
Empty arrays are perfectly acceptable for categories with no matches.

Example response format:
{{"Christmas Movies": [1, 17, 22], "Star Wars Saga": [4, 12, 15], "Halloween Movies": []}}"""
        
        # Build the user prompt with structured JSON payload
        data_payload = {
            "movies": movies_payload,
            "categories": categories_payload,
            "task": "classify_all_movies_across_all_categories"
        }
        
        # Convert to compact JSON string
        json_payload = json.dumps(data_payload, separators=(',', ':'))
        
        user_prompt = f"""DATA PAYLOAD:
{json_payload}

ORGANIZE MOVIES INTO COLLECTIONS USING MINIMAL INDEX FORMAT.

For each category, identify which movies belong in that collection by their index number (1, 2, 3, etc.).
Arrange franchise movies in chronological story order.

Respond with JSON in this exact minimal format:
{{
  "Christmas Movies": [1, 17, 22],
  "Star Wars Saga": [4, 12, 15, 18, 25],
  "Marvel Cinematic Universe": [2, 7, 11, 19, 23],
  "Halloween Movies": []
}}

RULES:
- Use movie indices only (numbers from 1 to {len(movies_payload)})
- No titles, no metadata, no confidence scores - just numbers
- Empty arrays [] for categories with no matching movies
- For franchises, order indices by chronological story timeline (not release order)

EXAMPLE: If movie #5 is "Die Hard" and it belongs in Christmas Movies, include 5 in that array.

IMPORTANT: Only include movies you're confident belong in each category. This ultra-minimal format saves ~95% of tokens."""

        return system_prompt, user_prompt
    
    def _parse_mega_batch_response(self, response: ChatCompletion, 
                                 media_items: List[Dict[str, Any]],
                                 categories: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Parse the minimal index-based mega-batch response into individual classification results."""
        all_results = []
        
        try:
            # Extract and parse JSON response
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")
            
            logger.debug(f"Index-based response content: {content[:500]}...")
            
            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Full response content: {content}")
                
                # Try to extract JSON from response if it's wrapped in text
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    extracted_json = json_match.group()
                    logger.info(f"Extracted JSON: {extracted_json[:500]}...")
                    try:
                        response_data = json.loads(extracted_json)
                    except json.JSONDecodeError as e2:
                        logger.error(f"Extracted JSON also invalid: {e2}")
                        raise ValueError(f"Invalid JSON in index response: {e}")
                else:
                    raise ValueError(f"No valid JSON found in index response: {e}")
            
            # Response should be a direct object with category names as keys and index arrays as values
            if not isinstance(response_data, dict):
                raise ValueError("Response is not a JSON object")
            
            # Create category name-to-category lookup
            name_to_category = {}
            for category in categories:
                name_to_category[category['name']] = category
            
            # Track which items have been matched for each category
            category_matches = {}
            for category in categories:
                category_matches[category['id']] = set()
            
            # Process each category's index array
            for category_name, indices in response_data.items():
                category = name_to_category.get(category_name)
                if not category:
                    logger.warning(f"Unknown category in response: {category_name}")
                    continue
                
                is_franchise = category.get('is_franchise', False)
                
                if not isinstance(indices, list):
                    logger.warning(f"Category '{category_name}' indices is not a list: {indices}")
                    continue
                
                # Process each index in the array
                for position, index in enumerate(indices, 1):
                    if not isinstance(index, int):
                        logger.warning(f"Invalid index in {category_name}: {index} (not integer)")
                        continue
                    
                    # Convert 1-based index to 0-based
                    item_idx = index - 1
                    
                    if item_idx < 0 or item_idx >= len(media_items):
                        logger.warning(f"Index {index} out of range for {category_name} (max: {len(media_items)})")
                        continue
                    
                    item = media_items[item_idx]
                    category_matches[category['id']].add(item['id'])
                    
                    # Create classification result
                    result = ClassificationResult(
                        media_item_id=item['id'],
                        category_id=category['id'],
                        matches=True,
                        confidence=0.8,  # Default confidence for index matches
                        reasoning=f"AI classified in {category_name} (index {index})",
                        tokens_used=response.usage.total_tokens // (len(media_items) * len(categories)),
                        processing_time=0.0
                    )
                    
                    # Add franchise-specific data if applicable
                    if is_franchise:
                        result.franchise_position = position  # Use position in chronological array
                        result.franchise_reasoning = f"Chronological position {position} in {category_name}"
                        # Note: We lose franchise_year info with minimal format, but gain efficiency
                    
                    all_results.append(result)
                    logger.debug(f"Index match: Movie #{index} ({item['title']}) → {category_name}")
            
            # Create non-match classifications for all items not matched in each category
            for category in categories:
                matched_item_ids = category_matches[category['id']]
                
                for item in media_items:
                    if item['id'] not in matched_item_ids:
                        result = ClassificationResult(
                            media_item_id=item['id'],
                            category_id=category['id'],
                            matches=False,
                            confidence=0.1,
                            reasoning=f"Not classified in {category['name']}",
                            tokens_used=response.usage.total_tokens // (len(media_items) * len(categories)),
                            processing_time=0.0
                        )
                        all_results.append(result)
            
            matches_count = sum(1 for r in all_results if r.matches)
            logger.info(f"Successfully parsed {len(all_results)} classification results from index-based response ({matches_count} matches)")
            
        except Exception as e:
            logger.error(f"Error parsing index-based response: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Return error results for all item-category combinations
            for item in media_items:
                for category in categories:
                    all_results.append(ClassificationResult(
                        media_item_id=item['id'],
                        category_id=category['id'],
                        matches=False,
                        error=f"Parse error: {str(e)}"
                    ))
        
        return all_results
    
    def _find_movie_by_title(self, target_title: str, media_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a movie in the media items list by title with fuzzy matching."""
        # Exact match first
        for item in media_items:
            if item['title'] == target_title:
                return item
        
        # Fuzzy matching - normalize titles for comparison
        target_normalized = target_title.lower().strip()
        
        for item in media_items:
            item_normalized = item['title'].lower().strip()
            
            # Check if target title is contained in item title or vice versa
            if target_normalized in item_normalized or item_normalized in target_normalized:
                return item
            
            # Check for common variations (remove year, punctuation, etc.)
            import re
            target_clean = re.sub(r'[^\w\s]', '', target_normalized)
            item_clean = re.sub(r'[^\w\s]', '', item_normalized)
            
            if target_clean in item_clean or item_clean in target_clean:
                return item
        
        logger.warning(f"Could not find movie: '{target_title}' in library")
        return None
    
    def test_api_connection(self) -> Dict[str, Any]:
        """Test the OpenAI API connection."""
        try:
            # Prepare test API call
            api_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello, this is a test. Please respond with 'Test successful'."}]
            }
            
            # Handle different model parameter requirements for test
            if self.model.startswith("o1"):
                # o1 models don't support max_tokens or temperature
                pass
            elif self.model.startswith("o4"):
                # o4 models use max_completion_tokens instead of max_tokens
                api_params["max_completion_tokens"] = 10
            else:
                # Standard models
                api_params["max_tokens"] = 10
                api_params["temperature"] = 0
            
            response = self.client.chat.completions.create(**api_params)
            
            return {
                "success": True,
                "model": self.model,
                "response": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "cost_estimate": (response.usage.total_tokens / 1000) * self.cost_per_1k_tokens
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# Global OpenAI client instance
_openai_client: Optional[OpenAIClient] = None

def get_openai_client() -> OpenAIClient:
    """Get the global OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client