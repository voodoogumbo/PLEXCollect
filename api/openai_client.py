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

@dataclass
class BatchResult:
    """Result of batch processing."""
    results: List[ClassificationResult]
    total_tokens: int
    total_cost: float
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    suggested_name: Optional[str] = None  # For natural language search results

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

        self.api_key = api_key or config.ai.api_key
        self.model = model or config.ai.model
        self.max_tokens = config.ai.max_tokens
        self.temperature = config.ai.temperature
        self.batch_size = config.ai.batch_size

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Rate limiter
        self.rate_limiter = RateLimiter(
            config.ai.rate_limit.requests_per_minute,
            config.ai.rate_limit.tokens_per_minute
        )

        # Cost tracking (approximate)
        self.cost_per_1k_tokens = self._get_cost_per_1k_tokens()

    def _get_cost_per_1k_tokens(self) -> float:
        """Get approximate cost per 1000 tokens for the model."""
        costs = [
            ("o4-mini", 0.001),
            ("o1-mini", 0.003),
            ("o1-preview", 0.015),
            ("gpt-4o-mini", 0.00015),
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
        return len(text) // 4 + 100

    def _build_classification_prompt(self, media_items: List[Dict[str, Any]],
                                   category: Dict[str, Any]) -> Tuple[str, str]:
        """Build the classification prompt for a batch of media items."""
        return self._build_regular_prompt(media_items, category)

    def _build_regular_prompt(self, media_items: List[Dict[str, Any]],
                            category: Dict[str, Any]) -> Tuple[str, str]:
        """Build prompt for collections."""

        system_prompt = f"""You are a movie and TV show classifier specializing in subjective, vibe-based categorization. Your task is to determine which items belong to the category "{category['name']}".

Category Description: {category['description']}
Classification Rule: {category['prompt']}

For each media item, respond with JSON containing:
- "matches": true/false if the item belongs to this category
- "confidence": 0.0-1.0 confidence score
- "reasoning": Brief explanation for your decision

These are subjective, taste-based categories. Use the movie's metadata (genres, summary, cast, rating) to judge whether it fits the vibe described.
Be consistent and thoughtful in your classifications."""

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
                pass
            elif self.model.startswith("o4"):
                api_params["max_completion_tokens"] = self.max_tokens
            else:
                api_params["max_tokens"] = self.max_tokens
                api_params["temperature"] = self.temperature
                api_params["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**api_params)

            processing_time = time.time() - start_time
            logger.debug(f"AI API call completed in {processing_time:.2f}s with model {self.model}")

            # Check for truncated response
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
            logger.error(f"AI API error: {e}")
            raise

    def _parse_classification_response(self, response: ChatCompletion,
                                     media_items: List[Dict[str, Any]],
                                     category_id: int) -> List[ClassificationResult]:
        """Parse the OpenAI response into classification results."""
        results = []

        try:
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from AI")

            logger.debug(f"AI response content: {content[:500]}...")

            try:
                response_data = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")

            if isinstance(response_data, dict) and "results" in response_data:
                classifications = response_data["results"]
            elif isinstance(response_data, list):
                classifications = response_data
            else:
                raise ValueError("Response does not contain 'results' array")

            if not isinstance(classifications, list):
                raise ValueError("Classifications is not a list")

            if len(classifications) != len(media_items):
                logger.warning(f"Mismatch: {len(classifications)} results for {len(media_items)} items")

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

                result = ClassificationResult(
                    media_item_id=item['id'],
                    category_id=category_id,
                    matches=classification.get('matches', False),
                    confidence=classification.get('confidence'),
                    reasoning=classification.get('reasoning'),
                    tokens_used=response.usage.total_tokens // len(media_items),
                    processing_time=0.0
                )

                results.append(result)

        except Exception as e:
            logger.error(f"Error parsing classification response: {e}")
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
            if len(media_items) > self.batch_size:
                media_items = media_items[:self.batch_size]
                logger.warning(f"Batch size limited to {self.batch_size} items")

            logger.info(f"Classifying {len(media_items)} items for category '{category['name']}'")

            system_prompt, user_prompt = self._build_classification_prompt(media_items, category)
            response = await self._make_api_call(system_prompt, user_prompt)
            results = self._parse_classification_response(response, media_items, category['id'])

            total_tokens = response.usage.total_tokens
            total_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            processing_time = time.time() - start_time

            for result in results:
                result.processing_time = processing_time / len(results)

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
            result = await self.classify_all_items_mega_batch(media_items, categories, progress_callback)
            logger.info(f"Mega-batch classification complete: {result.total_tokens} tokens, ${result.total_cost:.4f} cost")
            return [result]

        except Exception as e:
            logger.error(f"Mega-batch classification failed, falling back to old method: {e}")
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

            for i in range(0, len(media_items), self.batch_size):
                batch = media_items[i:i + self.batch_size]

                try:
                    result = await self.classify_media_batch(batch, category)
                    all_results.append(result)

                    completed_batches += 1

                    if progress_callback:
                        progress = completed_batches / total_batches
                        progress_callback(progress, f"Processed category '{category['name']}' batch {i//self.batch_size + 1}")

                    if i + self.batch_size < len(media_items):
                        await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"Failed to process batch for category {category['name']}: {e}")
                    continue

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

        MAX_MOVIES_PER_MEGA_BATCH = 40

        if len(media_items) > MAX_MOVIES_PER_MEGA_BATCH:
            logger.info(f"Large library detected ({len(media_items)} items). Using chunked mega-batch approach.")
            return await self._classify_chunked_mega_batch(media_items, categories, progress_callback)

        if progress_callback:
            progress_callback(0.2, "Building JSON mega-batch payload...")

        try:
            system_prompt, user_prompt = self._build_mega_batch_prompt(media_items, categories)

            if progress_callback:
                progress_callback(0.4, "Sending JSON mega-batch to AI...")

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

            all_results = self._parse_mega_batch_response(response, media_items, categories)

            if progress_callback:
                progress_callback(0.9, "Finalizing JSON mega-batch results...")

            total_tokens = response.usage.total_tokens
            total_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            processing_time = time.time() - start_time

            for result in all_results:
                result.processing_time = processing_time / len(all_results)

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

        chunks = [media_items[i:i + MAX_MOVIES_PER_CHUNK]
                 for i in range(0, len(media_items), MAX_MOVIES_PER_CHUNK)]

        logger.info(f"Processing {len(media_items)} items in {len(chunks)} mega-batch chunks")

        for chunk_idx, chunk in enumerate(chunks):
            if progress_callback:
                chunk_progress = (chunk_idx / len(chunks)) * 0.9
                progress_callback(chunk_progress, f"Processing mega-batch chunk {chunk_idx + 1}/{len(chunks)} ({len(chunk)} items)")

            try:
                chunk_result = await self.classify_all_items_mega_batch(chunk, categories)

                all_results.extend(chunk_result.results)
                total_tokens += chunk_result.total_tokens
                total_cost += chunk_result.total_cost

                logger.info(f"Chunk {chunk_idx + 1}/{len(chunks)} completed: {len(chunk_result.results)} classifications")

            except ValueError as e:
                if "truncated" in str(e).lower() and len(chunk) > 10:
                    logger.warning(f"Chunk {chunk_idx + 1} truncated with {len(chunk)} items, splitting into smaller chunks")
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
                for item in chunk:
                    for category in categories:
                        all_results.append(ClassificationResult(
                            media_item_id=item['id'],
                            category_id=category['id'],
                            matches=False,
                            error=f"Chunk processing failed: {str(e)}"
                        ))

        processing_time = time.time() - start_time

        for result in all_results:
            result.processing_time = processing_time / len(all_results) if all_results else 0

        successful_classifications = sum(1 for r in all_results if not r.error)
        matches = sum(1 for r in all_results if r.matches and not r.error)

        logger.info(f"Chunked mega-batch completed: {successful_classifications}/{len(all_results)} successful, "
                   f"{matches} matches, {total_tokens} tokens, ${total_cost:.4f}")

        if progress_callback:
            progress_callback(1.0, f"Chunked mega-batch complete! {matches} matches found")

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
                "summary": item.get('summary', '')[:400],
                "content_rating": item.get('content_rating'),
                "rating": item.get('rating')
            }
            movies_payload.append(movie_data)

        categories_payload = []
        for category in categories:
            category_data = {
                "id": category['id'],
                "name": category['name'],
                "description": category['description'],
                "classification_rule": category['prompt']
            }
            categories_payload.append(category_data)

        system_prompt = f"""You are Plex-Classifier v5, an AI-powered movie collection organizer specializing in subjective, vibe-based collections.

Your task: Classify movies into taste-based collections using minimal index arrays.

Classification Guidelines:
- These are subjective, taste-based categories. Use the movie's metadata (genres, summary, cast, rating) to judge whether it fits the vibe described.
- Use confidence >= 0.6 to include a movie in a collection
- Consider all metadata: title, year, genres, directors, summary, content rating
- Return movie indices only (numbers), not titles or details
- A movie can belong to multiple collections if it fits multiple vibes

CRITICAL: You must respond with ONLY valid JSON. No explanations, no markdown, no code blocks.
Response format: JSON object with category names as keys and arrays of movie indices as values.
Empty arrays are perfectly acceptable for categories with no matches.

Example response format:
{{"Cozy Comfort Movies": [1, 17, 22], "Mind-Bending Movies": [4, 12, 15], "Adrenaline Rush": []}}"""

        data_payload = {
            "movies": movies_payload,
            "categories": categories_payload,
            "task": "classify_all_movies_across_all_categories"
        }

        json_payload = json.dumps(data_payload, separators=(',', ':'))

        user_prompt = f"""DATA PAYLOAD:
{json_payload}

ORGANIZE MOVIES INTO COLLECTIONS USING MINIMAL INDEX FORMAT.

For each category, identify which movies belong in that collection by their index number (1, 2, 3, etc.).

Respond with JSON in this exact minimal format:
{{
  "Cozy Comfort Movies": [1, 17, 22],
  "Mind-Bending Movies": [4, 12, 15, 18, 25],
  "Adrenaline Rush": []
}}

RULES:
- Use movie indices only (numbers from 1 to {len(movies_payload)})
- No titles, no metadata, no confidence scores - just numbers
- Empty arrays [] for categories with no matching movies
- Consider the vibe/feel described in each category's classification rule

EXAMPLE: If movie #5 is "Die Hard" and it fits "Adrenaline Rush", include 5 in that array.

IMPORTANT: Only include movies you're confident belong in each category. This ultra-minimal format saves ~95% of tokens."""

        return system_prompt, user_prompt

    def _parse_mega_batch_response(self, response: ChatCompletion,
                                 media_items: List[Dict[str, Any]],
                                 categories: List[Dict[str, Any]]) -> List[ClassificationResult]:
        """Parse the minimal index-based mega-batch response into individual classification results."""
        all_results = []

        try:
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from AI")

            logger.debug(f"Index-based response content: {content[:500]}...")

            try:
                response_data = json.loads(content)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed: {e}")
                logger.error(f"Full response content: {content}")

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

            if not isinstance(response_data, dict):
                raise ValueError("Response is not a JSON object")

            name_to_category = {}
            for category in categories:
                name_to_category[category['name']] = category

            category_matches = {}
            for category in categories:
                category_matches[category['id']] = set()

            for category_name, indices in response_data.items():
                category = name_to_category.get(category_name)
                if not category:
                    logger.warning(f"Unknown category in response: {category_name}")
                    continue

                if not isinstance(indices, list):
                    logger.warning(f"Category '{category_name}' indices is not a list: {indices}")
                    continue

                for index in indices:
                    if not isinstance(index, int):
                        logger.warning(f"Invalid index in {category_name}: {index} (not integer)")
                        continue

                    item_idx = index - 1

                    if item_idx < 0 or item_idx >= len(media_items):
                        logger.warning(f"Index {index} out of range for {category_name} (max: {len(media_items)})")
                        continue

                    item = media_items[item_idx]
                    category_matches[category['id']].add(item['id'])

                    result = ClassificationResult(
                        media_item_id=item['id'],
                        category_id=category['id'],
                        matches=True,
                        confidence=0.8,
                        reasoning=f"AI classified in {category_name} (index {index})",
                        tokens_used=response.usage.total_tokens // (len(media_items) * len(categories)),
                        processing_time=0.0
                    )

                    all_results.append(result)
                    logger.debug(f"Index match: Movie #{index} ({item['title']}) -> {category_name}")

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

            for item in media_items:
                for category in categories:
                    all_results.append(ClassificationResult(
                        media_item_id=item['id'],
                        category_id=category['id'],
                        matches=False,
                        error=f"Parse error: {str(e)}"
                    ))

        return all_results

    # Natural Language Search

    def _build_natural_language_prompt(self, media_items: List[Dict[str, Any]],
                                      query: str) -> Tuple[str, str]:
        """Build prompt for natural language collection search."""
        movies_payload = []
        for i, item in enumerate(media_items, 1):
            movie_data = {
                "id": i,
                "media_item_id": item['id'],
                "title": item.get('title', 'Unknown'),
                "year": item.get('year'),
                "genres": item.get('genres', []),
                "summary": item.get('summary', '')[:300],
                "content_rating": item.get('content_rating'),
                "rating": item.get('rating')
            }
            movies_payload.append(movie_data)

        system_prompt = """You are a movie collection curator. A user wants to create a custom Plex collection using natural language.

Your task:
1. Interpret the user's query to understand what kind of movies they're looking for
2. Search through the provided movie list and find all matches
3. Suggest a short, catchy collection name

Respond with ONLY valid JSON:
{
  "collection_name": "Suggested Collection Name",
  "matching_indices": [1, 5, 12, ...],
  "reasoning": "Brief explanation of what you looked for"
}

RULES:
- Use movie indices (1-based numbers)
- Be generous but accurate — include movies that clearly fit
- The collection name should be concise (2-5 words)
- Empty array if nothing matches"""

        json_payload = json.dumps({"movies": movies_payload}, separators=(',', ':'))

        user_prompt = f"""USER QUERY: "{query}"

MOVIE LIBRARY:
{json_payload}

Find all movies matching the user's description and suggest a collection name."""

        return system_prompt, user_prompt

    async def natural_language_search(self, media_items: List[Dict[str, Any]],
                                     query: str,
                                     progress_callback: Optional[callable] = None) -> BatchResult:
        """Search library using natural language query and return matching items."""
        start_time = time.time()

        MAX_ITEMS_PER_CHUNK = 40

        if len(media_items) > MAX_ITEMS_PER_CHUNK:
            return await self._chunked_natural_language_search(media_items, query, progress_callback)

        if progress_callback:
            progress_callback(0.2, "Building natural language search query...")

        try:
            system_prompt, user_prompt = self._build_natural_language_prompt(media_items, query)

            if progress_callback:
                progress_callback(0.5, "Searching your library with AI...")

            response = await self._make_api_call(system_prompt, user_prompt)

            if progress_callback:
                progress_callback(0.8, "Parsing search results...")

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from AI")

            try:
                response_data = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    response_data = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")

            suggested_name = response_data.get("collection_name", "Custom Collection")
            matching_indices = response_data.get("matching_indices", [])

            results = []
            for index in matching_indices:
                if not isinstance(index, int):
                    continue
                item_idx = index - 1
                if 0 <= item_idx < len(media_items):
                    item = media_items[item_idx]
                    results.append(ClassificationResult(
                        media_item_id=item['id'],
                        category_id=0,  # No category yet
                        matches=True,
                        confidence=0.8,
                        reasoning=response_data.get("reasoning", "Matched natural language query")
                    ))

            total_tokens = response.usage.total_tokens
            total_cost = (total_tokens / 1000) * self.cost_per_1k_tokens
            processing_time = time.time() - start_time

            if progress_callback:
                progress_callback(1.0, f"Found {len(results)} matching movies!")

            return BatchResult(
                results=results,
                total_tokens=total_tokens,
                total_cost=total_cost,
                processing_time=processing_time,
                success=True,
                suggested_name=suggested_name
            )

        except Exception as e:
            logger.error(f"Natural language search failed: {e}")
            return BatchResult(
                results=[],
                total_tokens=0,
                total_cost=0.0,
                processing_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )

    async def _chunked_natural_language_search(self, media_items: List[Dict[str, Any]],
                                               query: str,
                                               progress_callback: Optional[callable] = None) -> BatchResult:
        """Process natural language search in chunks for large libraries."""
        MAX_ITEMS_PER_CHUNK = 40
        start_time = time.time()

        all_results = []
        total_tokens = 0
        total_cost = 0.0
        suggested_name = None

        chunks = [media_items[i:i + MAX_ITEMS_PER_CHUNK]
                 for i in range(0, len(media_items), MAX_ITEMS_PER_CHUNK)]

        logger.info(f"NL search: Processing {len(media_items)} items in {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            if progress_callback:
                chunk_progress = (chunk_idx / len(chunks)) * 0.9
                progress_callback(chunk_progress, f"Searching chunk {chunk_idx + 1}/{len(chunks)}...")

            try:
                chunk_result = await self.natural_language_search(chunk, query)
                all_results.extend(chunk_result.results)
                total_tokens += chunk_result.total_tokens
                total_cost += chunk_result.total_cost

                if suggested_name is None and chunk_result.suggested_name:
                    suggested_name = chunk_result.suggested_name
            except Exception as e:
                logger.error(f"NL search chunk {chunk_idx + 1} failed: {e}")

        processing_time = time.time() - start_time

        if progress_callback:
            progress_callback(1.0, f"Search complete! Found {len(all_results)} matches")

        return BatchResult(
            results=all_results,
            total_tokens=total_tokens,
            total_cost=total_cost,
            processing_time=processing_time,
            success=True,
            suggested_name=suggested_name or "Custom Collection"
        )

    def _find_movie_by_title(self, target_title: str, media_items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find a movie in the media items list by title with fuzzy matching."""
        for item in media_items:
            if item['title'] == target_title:
                return item

        target_normalized = target_title.lower().strip()

        for item in media_items:
            item_normalized = item['title'].lower().strip()

            if target_normalized in item_normalized or item_normalized in target_normalized:
                return item

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
            api_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hello, this is a test. Please respond with 'Test successful'."}]
            }

            if self.model.startswith("o1"):
                pass
            elif self.model.startswith("o4"):
                api_params["max_completion_tokens"] = 10
            else:
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
