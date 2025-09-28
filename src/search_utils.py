"""
Search and parsing utilities for the Agentic Ticker system.

This module consolidates all search and parsing related functionality to eliminate
code duplication across the codebase. It provides unified interfaces for:
- Web search functionality (DDGS integration)
- Search result parsing and validation
- Content extraction and cleaning
- Search query formatting and optimization
- Search error handling and fallback mechanisms
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import requests
from pydantic import BaseModel, Field

# Import decorators
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.decorators import handle_errors, log_execution, time_execution, validate_inputs, cache_result, retry_on_failure

# Configure logging
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Represents a standardized search result."""
    title: str = Field(default="", description="Title of the search result")
    href: str = Field(default="", description="URL of the search result")
    content: str = Field(
        default="", description="Main content/snippet of the search result"
    )
    source: str = Field(default="ddgs", description="Source of the search result")
    relevance_score: float = Field(
        default=0.0, description="Relevance score of the result"
    )


class SearchConfig(BaseModel):
    """Configuration for search operations."""
    max_results: int = Field(
        default=5, ge=1, le=20,
        description="Maximum number of results to return"
    )
    timeout: int = Field(
        default=30, ge=5, le=120,
        description="Timeout in seconds for search operations"
    )
    region: str = Field(default="us-en", description="Search region")
    safesearch: str = Field(default="moderate", description="Safe search level")
    retry_count: int = Field(
        default=2, ge=0, le=5,
        description="Number of retry attempts on failure"
    )
    enable_fallback: bool = Field(
        default=True, description="Enable fallback search methods"
    )


class SearchError(Exception):
    """Custom exception for search-related errors."""
    pass


class SearchUtils:
    """Main utility class for search and parsing operations."""

    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize SearchUtils with optional configuration."""
        self.config = config or SearchConfig()
        self._setup_gemini_config()

    def _setup_gemini_config(self) -> None:
        """Setup Gemini API configuration from environment variables."""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.gemini_api_base = os.getenv(
            "GEMINI_API_BASE",
            "https://generativelanguage.googleapis.com/v1beta"
        )

        if not self.gemini_api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables")

    @handle_errors(default_return=[], log_errors=True)
    @log_execution(include_args=False, include_result=False)
    @time_execution(log_threshold=5.0)
    @validate_inputs(query='non_empty_string')
    @retry_on_failure(max_attempts=3, delay=1.0, exceptions=(SearchError, requests.RequestException))
    def web_search(
        self, query: str, config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """
        Perform web search using DDGS (DuckDuckGo Search) library.

        Args:
            query: Search query string
            config: Optional search configuration override

        Returns:
            List of SearchResult objects

        Raises:
            SearchError: If search fails and no fallback is available
        """
        search_config = config or self.config

        try:
            return self._ddgs_search(query, search_config)
        except Exception as e:
            logger.error(f"DDGS search failed: {e}")
            if search_config.enable_fallback:
                return self._fallback_search(query, search_config)
            raise SearchError(f"Web search failed: {e}")

    def _ddgs_search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Internal DDGS search implementation."""
        try:
            from ddgs import DDGS

            # Initialize DDGS with specified settings
            ddgs = DDGS()

            # Perform text search
            results = ddgs.text(
                query,
                region=config.region,
                safesearch=config.safesearch,
                max_results=config.max_results
            )

            # Convert DDGS results to standardized format
            formatted_results = []
            for result in results:
                search_result = SearchResult(
                    title=result.get('title', ''),
                    href=result.get('href', ''),
                    content=result.get('body', ''),
                    source='ddgs'
                )
                formatted_results.append(search_result)

            logger.info(
                f"Web search returned {len(formatted_results)} results "
                f"for query: {query}"
            )
            return formatted_results[:config.max_results]

        except ImportError:
            raise SearchError(
                "DDGS library not available. Install with: pip install ddgs"
            )
        except Exception as e:
            raise SearchError(f"DDGS search error: {e}")

    def _fallback_search(self, query: str, config: SearchConfig) -> List[SearchResult]:
        """Fallback search method when DDGS is not available."""
        logger.warning("Using fallback search method")

        # For now, return empty results - could be extended with other search
        # providers
        return []

    def extract_search_text(self, results: List[SearchResult]) -> str:
        """
        Extract and combine text content from search results.

        Args:
            results: List of SearchResult objects

        Returns:
            Combined text content from all results
        """
        if not results:
            return ""

        # Combine title and content from all results
        text_parts = []
        for result in results:
            combined_text = f"{result.title} {result.content}".strip()
            if combined_text:
                text_parts.append(combined_text)

        return " ".join(text_parts)

    def clean_text(self, text: str, remove_special_chars: bool = True) -> str:
        """
        Clean and normalize text content.

        Args:
            text: Input text to clean
            remove_special_chars: Whether to remove special characters

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Basic cleaning
        cleaned = text.strip()

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # Remove special characters if requested
        if remove_special_chars:
            cleaned = re.sub(r'[^\w\s\-.,;:!?]', '', cleaned)

        return cleaned

    def format_search_query(self, base_query: str, query_type: str = "general") -> str:
        """
        Format and optimize search queries based on type.

        Args:
            base_query: Base search query
            query_type: Type of query ('general', 'ticker', 'crypto', 'company')

        Returns:
            Formatted search query
        """
        query = base_query.strip()

        # Add context based on query type
        if query_type == "ticker":
            query = f"{query} stock ticker symbol"
        elif query_type == "crypto":
            query = f"{query} cryptocurrency ticker symbol"
        elif query_type == "company":
            query = f"{query} company information stock"
        elif query_type == "crypto_id":
            query = f"{query} CoinGecko coin ID cryptocurrency"

        return query

    def gemini_api_call(self, prompt: str, temperature: float = 0.1) -> str:
        """
        Make a standardized API call to Gemini.

        Args:
            prompt: Prompt text for Gemini
            temperature: Temperature parameter for generation

        Returns:
            Gemini response text

        Raises:
            SearchError: If API call fails
        """
        if not self.gemini_api_key:
            raise SearchError("Gemini API key not configured")

        url = (
            f"{self.gemini_api_base}/models/{self.gemini_model}:"
            f"generateContent?key={self.gemini_api_key}"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "text/plain"
            }
        }

        try:
            response = requests.post(url, json=body, timeout=self.config.timeout)
            response.raise_for_status()
            data = response.json()

            # Extract text from response
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text

        except requests.RequestException as e:
            raise SearchError(f"Gemini API request failed: {e}")
        except (KeyError, IndexError) as e:
            raise SearchError(f"Gemini API response parsing failed: {e}")

    def parse_ticker_from_search(
        self, search_results: List[SearchResult], original_input: str
    ) -> str:
        """
        Parse and extract ticker symbol from search results using Gemini.

        Args:
            search_results: List of search results
            original_input: Original user input

        Returns:
            Parsed ticker symbol or empty string if parsing fails
        """
        if not search_results:
            return ""

        try:
            search_text = self.extract_search_text(search_results)

            prompt = f"""
            Based on these search results about "{original_input}", extract the
            correct stock ticker symbol.

            Search results:
            {search_text}

            Return ONLY the ticker symbol in uppercase, nothing else. No explanations.
            Examples: "AAPL", "MSFT", "BRK-A", "GOOGL"
            """

            ticker = self.gemini_api_call(prompt)

            # Clean up ticker (remove spaces, but keep hyphens for tickers like BRK-A)
            ticker = re.sub(r'[^A-Z0-9-]', '', ticker.upper())

            # Validate ticker format
            if re.match(r'^[A-Z0-9]{1,5}(-[A-Z0-9]{1,2})?$', ticker):
                return ticker

            return ""

        except Exception as e:
            logger.error(f"Failed to parse ticker from search results: {e}")
            return ""

    def parse_crypto_ticker_from_search(
        self, search_results: List[SearchResult], original_input: str
    ) -> str:
        """
        Parse and extract cryptocurrency ticker from search results using Gemini.

        Args:
            search_results: List of search results
            original_input: Original user input

        Returns:
            Parsed crypto ticker symbol or empty string if parsing fails
        """
        if not search_results:
            return ""

        try:
            search_text = self.extract_search_text(search_results)

            prompt = f"""
            Based on these search results about "{original_input}", extract the
            correct cryptocurrency ticker symbol.

            Search results:
            {search_text}

            Return ONLY the ticker symbol in uppercase, nothing else. No explanations.
            Examples: "BTC", "ETH", "XRP", "ADA"
            """

            ticker = self.gemini_api_call(prompt)

            # Clean up ticker (remove spaces, special characters except alphanumeric)
            ticker = re.sub(r'[^A-Z0-9]', '', ticker.upper())

            return ticker

        except Exception as e:
            logger.error(f"Failed to parse crypto ticker from search results: {e}")
            return ""

    def parse_coingecko_id_from_search(
        self, search_results: List[SearchResult], original_input: str
    ) -> str:
        """
        Parse and extract CoinGecko coin ID from search results using Gemini.

        Args:
            search_results: List of search results
            original_input: Original user input

        Returns:
            Parsed CoinGecko coin ID or empty string if parsing fails
        """
        if not search_results:
            return ""

        try:
            search_text = self.extract_search_text(search_results)

            prompt = f"""
            Based on these search results about "{original_input}", extract the
            correct CoinGecko coin ID.

            Search results:
            {search_text}

            Return ONLY the CoinGecko coin ID in lowercase, nothing else.
            No explanations.
            Examples: "bitcoin", "ethereum", "dogecoin", "phala-network"
            """

            coin_id = self.gemini_api_call(prompt)

            # Clean up coin ID (remove spaces, special characters except hyphens)
            coin_id = re.sub(r'[^a-z0-9-]', '', coin_id.lower())

            return coin_id

        except Exception as e:
            logger.error(f"Failed to parse CoinGecko ID from search results: {e}")
            return ""

    def classify_asset_type(self, input_text: str) -> str:
        """
        Classify asset type using Gemini API.

        Args:
            input_text: User input text to classify

        Returns:
            'stock', 'crypto', or 'ambiguous'
        """
        if not input_text or not input_text.strip():
            return "ambiguous"

        try:
            prompt = f"""
            Analyze the following input and determine if it refers to a stock,
            cryptocurrency, or is ambiguous:

            Input: "{input_text}"

            Classification rules:
            - Return "stock" if the input clearly refers to a traditional stock,
              company, or stock ticker symbol (e.g., "AAPL", "Apple", "Microsoft",
              "GOOGL")
            - Return "crypto" if the input clearly refers to a cryptocurrency,
              crypto token, or crypto exchange (e.g., "Bitcoin", "BTC",
              "Ethereum", "ETH", "Dogecoin")
            - Return "ambiguous" if the input could refer to both, is unclear,
              or doesn't clearly match either category

            Return ONLY one word: "stock", "crypto", or "ambiguous".
            No explanations, no formatting.

            Examples:
            - "AAPL" -> "stock"
            - "Apple" -> "stock"
            - "Bitcoin" -> "crypto"
            - "BTC" -> "crypto"
            - "Tesla" -> "stock"
            - "Ethereum" -> "crypto"
            - "TSLA" -> "stock"
            - "ETH" -> "crypto"
            - "Something random" -> "ambiguous"
            - "XYZ" -> "ambiguous"
            """

            classification = self.gemini_api_call(prompt).lower()

            # Validate the response
            if classification in ["stock", "crypto", "ambiguous"]:
                return classification

            return "ambiguous"

        except Exception as e:
            logger.error(f"Asset classification failed: {e}")
            return "ambiguous"

    def validate_and_clean_ticker(self, ticker: str) -> str:
        """
        Validate and clean ticker symbol.

        Args:
            ticker: Raw ticker string

        Returns:
            Cleaned and validated ticker symbol
        """
        if not ticker:
            return ""

        # Clean and normalize
        cleaned = ticker.strip().upper()

        # Remove special characters except hyphens
        cleaned = re.sub(r'[^A-Z0-9-]', '', cleaned)

        return cleaned

    def search_with_retry(
        self, query: str, query_type: str = "general",
        max_retries: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Perform search with retry logic.

        Args:
            query: Search query
            query_type: Type of query for formatting
            max_retries: Maximum number of retry attempts

        Returns:
            List of search results
        """
        retries = max_retries or self.config.retry_count
        formatted_query = self.format_search_query(query, query_type)

        for attempt in range(retries + 1):
            try:
                results = self.web_search(formatted_query)
                if results:
                    return results

                logger.warning(f"Search attempt {attempt + 1} returned no results")

            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")

                if attempt == retries:
                    logger.error(f"All search attempts failed for query: {query}")
                    return []

        return []

    def get_search_stats(self) -> Dict[str, Any]:
        """
        Get statistics about search operations.

        Returns:
            Dictionary with search statistics
        """
        return {
            "config": self.config.dict(),
            "gemini_configured": bool(self.gemini_api_key),
            "timestamp": datetime.now().isoformat()
        }


# Convenience functions for backward compatibility
def ddgs_search(query: str, max_results: int = 3, **kwargs) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility.

    Args:
        query: Search query
        max_results: Maximum number of results
        **kwargs: Additional arguments

    Returns:
        List of search result dictionaries
    """
    config = SearchConfig(max_results=max_results)
    search_utils = SearchUtils(config)

    try:
        results = search_utils.web_search(query)
        # Convert to legacy format
        return [
            {
                'title': result.title,
                'href': result.href,
                'content': result.content
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Legacy ddgs_search failed: {e}")
        return []


def extract_search_text(results: List[Dict[str, Any]]) -> str:
    """
    Legacy function for extracting search text.

    Args:
        results: List of search result dictionaries

    Returns:
        Combined text content
    """
    if not results:
        return ""

    text_parts = []
    for result in results:
        combined_text = f"{result.get('title', '')} {result.get('content', '')}".strip()
        if combined_text:
            text_parts.append(combined_text)

    return " ".join(text_parts)


# Global instance for convenience
_default_search_utils = SearchUtils()
