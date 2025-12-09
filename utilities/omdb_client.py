"""
OMDB API Client
===============

Client for interacting with the OMDB API.
"""

import requests
import logging
import time
from typing import Optional, Dict, Any, List
from urllib.parse import urlencode

from utilities.omdb_models import OMDBMovie

logger = logging.getLogger(__name__)

class OMDBError(Exception):
    """Base exception for OMDB API errors"""
    pass

class OMDBAPIKeyError(OMDBError):
    """Raised when API key is invalid or missing"""
    pass

class OMDBNotFoundError(OMDBError):
    """Raised when movie is not found"""
    pass

class OMDBRateLimitError(OMDBError):
    """Raised when rate limit is exceeded"""
    pass

class OMDBClient:
    """
    Client for OMDB API interactions.
    
    Usage:
        client = OMDBClient(api_key="your_key")
        movie = client.get_movie_by_title("Dumb and Dumber")
        print(movie.plot)
    """
    
    def __init__(
        self,
        api_key: str = "83fd90a1",
        base_url: str = "http://www.omdbapi.com/",
        timeout: int = 10,
        max_retries: int = 3
    ):
        """
        Initialize OMDB client.
        
        Args:
            api_key: OMDB API key (default: free tier key)
            base_url: OMDB API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        logger.info(f"Initialized OMDB client with API key: {api_key[:4]}...")
    
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to OMDB API with retry logic.
        
        Args:
            params: Query parameters
            
        Returns:
            API response as dictionary
            
        Raises:
            OMDBAPIKeyError: Invalid API key
            OMDBNotFoundError: Movie not found
            OMDBRateLimitError: Rate limit exceeded
            OMDBError: Other API errors
        """
        # Add API key to params
        params['apikey'] = self.api_key
        
        url = f"{self.base_url}/?{urlencode(params)}"
        
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Making OMDB API request (attempt {attempt + 1}/{self.max_retries}): {params}")
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API-level errors
                if data.get('Response') == 'False':
                    error_msg = data.get('Error', 'Unknown error')
                    
                    if 'Invalid API key' in error_msg:
                        raise OMDBAPIKeyError(f"Invalid OMDB API key: {error_msg}")
                    elif 'Movie not found' in error_msg or 'not found' in error_msg.lower():
                        raise OMDBNotFoundError(f"Movie not found: {error_msg}")
                    elif 'limit' in error_msg.lower():
                        raise OMDBRateLimitError(f"Rate limit exceeded: {error_msg}")
                    else:
                        raise OMDBError(f"OMDB API error: {error_msg}")
                
                logger.info(f"Successfully retrieved data for: {data.get('Title', 'unknown')}")
                return data
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise OMDBError("Request timed out after multiple retries")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise OMDBError(f"Request failed: {e}")
    
    def get_movie_by_title(
        self,
        title: str,
        plot: str = "full",
        year: Optional[int] = None
    ) -> OMDBMovie:
        """
        Get movie data by title.
        
        Args:
            title: Movie title to search for
            plot: Plot length ("short" or "full", default: "full")
            year: Optional year to narrow search
            
        Returns:
            OMDBMovie object
            
        Raises:
            OMDBNotFoundError: Movie not found
            OMDBError: Other API errors
            
        Example:
            movie = client.get_movie_by_title("Dumb and Dumber")
            print(movie.plot)
        """
        params = {
            't': title,
            'plot': plot
        }
        
        if year:
            params['y'] = str(year)
        
        data = self._make_request(params)
        return OMDBMovie.from_dict(data)
    
    def get_movie_by_imdb_id(
        self,
        imdb_id: str,
        plot: str = "full"
    ) -> OMDBMovie:
        """
        Get movie data by IMDb ID.
        
        Args:
            imdb_id: IMDb ID (e.g., "tt0109686")
            plot: Plot length ("short" or "full", default: "full")
            
        Returns:
            OMDBMovie object
            
        Raises:
            OMDBNotFoundError: Movie not found
            OMDBError: Other API errors
            
        Example:
            movie = client.get_movie_by_imdb_id("tt0109686")
            print(movie.title)
        """
        params = {
            'i': imdb_id,
            'plot': plot
        }
        
        data = self._make_request(params)
        return OMDBMovie.from_dict(data)
    
    def search_movies(
        self,
        query: str,
        year: Optional[int] = None,
        media_type: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Search for movies by title (returns multiple results).
        
        Args:
            query: Search query
            year: Optional year filter
            media_type: Optional type filter ("movie", "series", "episode")
            page: Page number (default: 1)
            
        Returns:
            List of movie dictionaries (not full OMDBMovie objects)
            
        Raises:
            OMDBNotFoundError: No results found
            OMDBError: API errors
            
        Example:
            results = client.search_movies("Star Wars")
            for result in results:
                print(result['Title'], result['Year'])
        """
        params = {
            's': query,
            'page': str(page)
        }
        
        if year:
            params['y'] = str(year)
        
        if media_type:
            params['type'] = media_type
        
        data = self._make_request(params)
        
        # Search results have different structure
        search_results = data.get('Search', [])
        total_results = data.get('totalResults', '0')
        
        logger.info(f"Found {total_results} results for query: {query}")
        return search_results
    
    def get_movie_by_title_with_fallback(
        self,
        title: str,
        plot: str = "full"
    ) -> Optional[OMDBMovie]:
        """
        Get movie by title with graceful fallback (returns None instead of raising).
        
        Args:
            title: Movie title
            plot: Plot length
            
        Returns:
            OMDBMovie object or None if not found
            
        Example:
            movie = client.get_movie_by_title_with_fallback("Nonexistent Movie")
            if movie:
                print(movie.plot)
            else:
                print("Movie not found")
        """
        try:
            return self.get_movie_by_title(title, plot=plot)
        except OMDBNotFoundError:
            logger.warning(f"Movie not found: {title}")
            return None
        except OMDBError as e:
            logger.error(f"Error fetching movie '{title}': {e}")
            return None
