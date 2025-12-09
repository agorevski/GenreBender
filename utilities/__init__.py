"""
OMDB API Utilities
==================

Utilities for fetching movie metadata from the OMDB API.

Usage:
    from utilities.omdb_client import OMDBClient
    from utilities.omdb_models import OMDBMovie
    
    client = OMDBClient()
    movie = client.get_movie_by_title("Dumb and Dumber")
    print(movie.plot)
"""

from utilities.omdb_models import OMDBMovie, OMDBRating
from utilities.omdb_client import OMDBClient
from utilities.omdb_cache import OMDBCache

__all__ = ['OMDBMovie', 'OMDBRating', 'OMDBClient', 'OMDBCache']
