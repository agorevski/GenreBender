#!/usr/bin/env python3
"""
OMDB API Example Usage
======================

Demonstrates how to use the OMDB API utilities.

Usage:
    python utilities/example_omdb.py
"""

import sys
import os
import logging

# Add project root to path so utilities can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import OMDBClient, OMDBCache
from utilities.omdb_client import OMDBNotFoundError, OMDBError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def example_basic_usage():
    """Demonstrate basic OMDB API usage.

    Creates an OMDB client and fetches movie details for 'Dumb and Dumber',
    displaying key information including title, year, genre, director, actors,
    IMDb rating, runtime, and plot.

    Raises:
        OMDBError: If the API request fails.
        OMDBNotFoundError: If the movie is not found.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Usage")
    print("="*70)
    
    client = OMDBClient()
    
    # Fetch a movie
    print("\nFetching 'Dumb and Dumber'...")
    movie = client.get_movie_by_title("Dumb and Dumber")
    
    print(f"\nTitle: {movie.title}")
    print(f"Year: {movie.year}")
    print(f"Genre: {movie.genre}")
    print(f"Director: {movie.director}")
    print(f"Actors: {movie.actors}")
    print(f"IMDb Rating: {movie.imdb_rating}")
    print(f"Runtime: {movie.runtime}")
    print(f"\nPlot:\n{movie.plot}")

def example_with_cache():
    """Demonstrate the OMDB cache system.

    Shows how to use the OMDBCache class to cache API responses. The first
    call fetches from the API while subsequent calls use the cached data.
    Also displays cache statistics.

    Raises:
        OMDBError: If the API request fails.
        OMDBNotFoundError: If the movie is not found.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Using Cache")
    print("="*70)
    
    client = OMDBClient()
    cache = OMDBCache(output_dir="outputs/example_movie")
    
    # First call - fetches from API
    print("\nFirst call (will fetch from API)...")
    movie = cache.get_or_fetch("The Matrix", client.get_movie_by_title)
    print(f"Retrieved: {movie.title} ({movie.year})")
    
    # Second call - uses cache
    print("\nSecond call (will use cache)...")
    movie_cached = cache.get_or_fetch("The Matrix", client.get_movie_by_title)
    print(f"Retrieved: {movie_cached.title} ({movie_cached.year})")
    
    # Show cache stats
    stats = cache.get_cache_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache directory: {stats['cache_dir']}")
    print(f"  Valid entries: {stats['valid_entries']}")
    print(f"  Expired entries: {stats['expired_entries']}")

def example_helper_properties():
    """Demonstrate OMDBMovie helper properties.

    Shows how to use convenience properties on the OMDBMovie model such as
    genres_list, actors_list, languages_list, runtime_minutes, imdb_rating_float,
    metascore_int, and get_rating_by_source().

    Raises:
        OMDBError: If the API request fails.
        OMDBNotFoundError: If the movie is not found.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Helper Properties")
    print("="*70)
    
    client = OMDBClient()
    movie = client.get_movie_by_title("Inception")
    
    print(f"\nMovie: {movie.title}")
    print(f"\nGenres (as list): {movie.genres_list}")
    print(f"Actors (as list): {movie.actors_list}")
    print(f"Languages (as list): {movie.languages_list}")
    print(f"\nRuntime (minutes): {movie.runtime_minutes}")
    print(f"IMDb Rating (float): {movie.imdb_rating_float}")
    print(f"Metascore (int): {movie.metascore_int}")
    
    # Get specific rating sources
    print(f"\nRatings:")
    for rating in movie.ratings:
        print(f"  {rating.source}: {rating.value}")
    
    rt_rating = movie.get_rating_by_source("Rotten Tomatoes")
    print(f"\nRotten Tomatoes rating: {rt_rating}")

def example_search():
    """Demonstrate movie search functionality.

    Shows how to search for movies by title using the search_movies method,
    which returns a list of matching results from the OMDB API.

    Raises:
        OMDBError: If the API request fails.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Searching Movies")
    print("="*70)
    
    client = OMDBClient()
    
    # Search for multiple movies
    print("\nSearching for 'Star Wars'...")
    results = client.search_movies("Star Wars")
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results[:5], 1):  # Show first 5
        print(f"  {i}. {result['Title']} ({result['Year']}) - {result['Type']}")

def example_search_by_imdb_id():
    """Demonstrate fetching a movie by IMDb ID.

    Shows how to retrieve movie details using a specific IMDb ID instead
    of searching by title.

    Raises:
        OMDBError: If the API request fails.
        OMDBNotFoundError: If the movie is not found.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Search by IMDb ID")
    print("="*70)
    
    client = OMDBClient()
    
    # Search by IMDb ID
    print("\nFetching movie by IMDb ID: tt0109686")
    movie = client.get_movie_by_imdb_id("tt0109686")
    print(f"Found: {movie.title} ({movie.year})")

def example_error_handling():
    """Demonstrate error handling patterns.

    Shows how to handle OMDBNotFoundError when a movie doesn't exist, and
    how to use the fallback method that returns None instead of raising
    an exception.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Error Handling")
    print("="*70)
    
    client = OMDBClient()
    
    # Try to fetch non-existent movie
    print("\nTrying to fetch non-existent movie...")
    try:
        movie = client.get_movie_by_title("XYZ123NONEXISTENT")
    except OMDBNotFoundError:
        print("✓ Caught OMDBNotFoundError: Movie not found")
    
    # Using fallback method
    print("\nUsing fallback method (returns None instead of raising)...")
    movie = client.get_movie_by_title_with_fallback("ANOTHERNONEXISTENT")
    if movie is None:
        print("✓ Gracefully handled: Movie returned None")

def example_year_filter():
    """Demonstrate filtering movies by year.

    Shows how to use the year parameter to distinguish between movies with
    the same title released in different years, using 'Dune' (1984 vs 2021)
    as an example.

    Raises:
        OMDBError: If the API request fails.
        OMDBNotFoundError: If the movie is not found.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: Year Filter")
    print("="*70)
    
    client = OMDBClient()
    
    # There are multiple "Dune" movies
    print("\nFetching 'Dune' (1984)...")
    dune_1984 = client.get_movie_by_title("Dune", year=1984)
    print(f"Found: {dune_1984.title} ({dune_1984.year})")
    print(f"Director: {dune_1984.director}")
    
    print("\nFetching 'Dune' (2021)...")
    dune_2021 = client.get_movie_by_title("Dune", year=2021)
    print(f"Found: {dune_2021.title} ({dune_2021.year})")
    print(f"Director: {dune_2021.director}")

def example_data_conversion():
    """Demonstrate data conversion between OMDBMovie and dictionary.

    Shows how to convert an OMDBMovie object to a dictionary using to_dict()
    and reconstruct it from a dictionary using OMDBMovie.from_dict().

    Raises:
        OMDBError: If the API request fails.
        OMDBNotFoundError: If the movie is not found.
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: Data Conversion")
    print("="*70)
    
    client = OMDBClient()
    movie = client.get_movie_by_title("The Shawshank Redemption")
    
    # Convert to dictionary
    movie_dict = movie.to_dict()
    print(f"\nConverted to dictionary:")
    print(f"  Keys: {', '.join(list(movie_dict.keys())[:5])}... (showing first 5)")
    
    # Reconstruct from dictionary
    from utilities.omdb_models import OMDBMovie
    reconstructed = OMDBMovie.from_dict(movie_dict)
    print(f"\nReconstructed from dictionary:")
    print(f"  Title: {reconstructed.title}")
    print(f"  Year: {reconstructed.year}")

def main():
    """Run all OMDB API example demonstrations.

    Executes each example function in sequence to demonstrate various
    features of the OMDB API utilities. Catches and displays any OMDB
    or unexpected errors that occur during execution.
    """
    print("\n" + "="*70)
    print("OMDB API UTILITIES - EXAMPLE USAGE")
    print("="*70)
    
    try:
        example_basic_usage()
        example_with_cache()
        example_helper_properties()
        example_search()
        example_search_by_imdb_id()
        example_error_handling()
        example_year_filter()
        example_data_conversion()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nFor more information, see utilities/README.md")
        
    except OMDBError as e:
        print(f"\n❌ OMDB Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
