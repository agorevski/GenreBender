# OMDB API Utilities

Utilities for fetching movie metadata from the [OMDB API](http://www.omdbapi.com/) (Open Movie Database).

## Overview

This module provides a Python interface to the OMDB API with built-in caching, error handling, and structured data models. It's designed as a standalone utility module that can be used independently or integrated into the GenreBender pipeline.

## Features

- ✅ **Structured Data Models**: Type-safe `OMDBMovie` and `OMDBRating` classes
- ✅ **Smart Caching**: Automatic caching with 30-day TTL to minimize API calls
- ✅ **Error Handling**: Comprehensive exception handling with custom error types
- ✅ **Retry Logic**: Automatic retry on network failures
- ✅ **Free API Key Included**: Pre-configured with free tier key (1,000 requests/day)
- ✅ **Multiple Search Methods**: Search by title, IMDb ID, or general query

## Installation

No additional dependencies required! The module uses libraries already in `requirements.txt`:
- `requests` (HTTP requests)
- `pathlib` (file operations)
- Standard library modules

## Quick Start

### Basic Usage

```python
from utilities.omdb_client import OMDBClient

# Initialize client (uses default API key from settings.yaml)
client = OMDBClient()

# Fetch movie by title
movie = client.get_movie_by_title("Dumb and Dumber")

# Access movie data
print(f"Title: {movie.title}")
print(f"Year: {movie.year}")
print(f"Genre: {movie.genre}")
print(f"Plot: {movie.plot}")
print(f"IMDb Rating: {movie.imdb_rating}")
print(f"Director: {movie.director}")
print(f"Actors: {movie.actors}")
```

### Using Cache

```python
from utilities.omdb_client import OMDBClient
from utilities.omdb_cache import OMDBCache

# Initialize client and cache
client = OMDBClient()
cache = OMDBCache(output_dir="outputs/my_movie")

# Fetch with automatic caching
movie = cache.get_or_fetch("Dumb and Dumber", client.get_movie_by_title)

# Subsequent calls will use cached data (no API call)
movie_again = cache.get_or_fetch("Dumb and Dumber", client.get_movie_by_title)
```

### Advanced Usage

```python
from utilities import OMDBClient, OMDBCache
from utilities.omdb_client import OMDBNotFoundError

client = OMDBClient()

# Search by IMDb ID
movie = client.get_movie_by_imdb_id("tt0109686")

# Search with year filter
movie = client.get_movie_by_title("Dune", year=2021)

# Search for multiple movies
results = client.search_movies("Star Wars")
for result in results:
    print(f"{result['Title']} ({result['Year']})")

# Graceful error handling
movie = client.get_movie_by_title_with_fallback("Nonexistent Movie")
if movie:
    print(movie.title)
else:
    print("Movie not found")

# Manual error handling
try:
    movie = client.get_movie_by_title("XYZ123")
except OMDBNotFoundError:
    print("Movie not found in OMDB database")
```

## Data Model

### OMDBMovie

The `OMDBMovie` class represents complete movie metadata:

```python
@dataclass
class OMDBMovie:
    title: str              # Movie title
    year: str               # Release year
    rated: str              # Content rating (PG-13, R, etc.)
    released: str           # Release date
    runtime: str            # Duration (e.g., "107 min")
    genre: str              # Comma-separated genres
    director: str           # Director name(s)
    writer: str             # Writer name(s)
    actors: str             # Comma-separated actors
    plot: str               # Full plot description
    language: str           # Comma-separated languages
    country: str            # Country of origin
    awards: str             # Awards and nominations
    poster_url: str         # Poster image URL
    ratings: List[OMDBRating]  # Ratings from various sources
    metascore: str          # Metascore rating
    imdb_rating: str        # IMDb rating
    imdb_votes: str         # Number of IMDb votes
    imdb_id: str            # IMDb ID (e.g., tt0109686)
    type: str               # Media type (movie, series, episode)
    dvd: str                # DVD release date
    box_office: str         # Box office earnings
    production: str         # Production company
    website: str            # Official website
```

### Helper Properties

```python
movie = client.get_movie_by_title("Dumb and Dumber")

# Get genres as list
genres = movie.genres_list  # ['Comedy']

# Get actors as list
actors = movie.actors_list  # ['Jim Carrey', 'Jeff Daniels', 'Lauren Holly']

# Get runtime as integer
runtime = movie.runtime_minutes  # 107

# Get IMDb rating as float
rating = movie.imdb_rating_float  # 7.3

# Get specific rating source
rt_rating = movie.get_rating_by_source("Rotten Tomatoes")  # '69%'
```

## Cache System

### Cache Configuration

The cache system stores OMDB responses in `outputs/<movie_name>/cache/omdb/` to avoid repeated API calls.

```python
from utilities.omdb_cache import OMDBCache

# Initialize with custom settings
cache = OMDBCache(
    output_dir="outputs/my_movie",    # Base directory
    cache_subdir="cache/omdb",         # Cache subdirectory
    ttl_seconds=2592000                # 30 days
)
```

### Cache Operations

```python
# Get from cache (returns None if not cached)
movie = cache.get("Dumb and Dumber")

# Set cache manually
cache.set("Dumb and Dumber", movie)

# Get or fetch (recommended)
movie = cache.get_or_fetch("Dumb and Dumber", client.get_movie_by_title)

# Invalidate specific entry
cache.invalidate("Dumb and Dumber")

# Clear all cache
cache.clear_all()

# Get cache statistics
stats = cache.get_cache_stats()
print(f"Valid entries: {stats['valid_entries']}")
print(f"Expired entries: {stats['expired_entries']}")
```

## Configuration

Settings are stored in `trailer_generator/config/settings.yaml`:

```yaml
omdb:
  api_key: "83fd90a1"  # Free tier (1,000 requests/day)
  base_url: "http://www.omdbapi.com/"
  cache_enabled: true
  cache_ttl: 2592000  # 30 days in seconds
  timeout: 10
  max_retries: 3
```

## Error Handling

The module provides specific exception types for different error scenarios:

```python
from utilities.omdb_client import (
    OMDBError,           # Base exception
    OMDBAPIKeyError,     # Invalid API key
    OMDBNotFoundError,   # Movie not found
    OMDBRateLimitError   # Rate limit exceeded
)

try:
    movie = client.get_movie_by_title("Some Movie")
except OMDBAPIKeyError:
    print("Invalid API key")
except OMDBNotFoundError:
    print("Movie not found")
except OMDBRateLimitError:
    print("Rate limit exceeded (1,000 requests/day)")
except OMDBError as e:
    print(f"Other OMDB error: {e}")
```

## API Key

### Free Tier
The module comes pre-configured with a free tier API key (`83fd90a1`) that allows:
- **1,000 requests per day**
- Full movie metadata
- Search functionality

### Getting Your Own Key
To get your own API key (free):
1. Visit [http://www.omdbapi.com/apikey.aspx](http://www.omdbapi.com/apikey.aspx)
2. Select "FREE! (1,000 daily limit)"
3. Enter your email and verify
4. Update `api_key` in `settings.yaml` or pass to `OMDBClient(api_key="your_key")`

### Paid Tiers
For higher limits, OMDB offers paid tiers:
- **Poster API**: 100,000 requests/day ($1/month)
- **Commercial**: 50,000,000 requests/year ($10/month)

## Example Script

See `utilities/example_omdb.py` for a complete working example:

```bash
python utilities/example_omdb.py
```

## Integration Ideas

The OMDB utilities can enhance the GenreBender pipeline:

### Stage 8: Narrative Generation
```python
# Enrich GPT-4 prompts with official metadata
from utilities import OMDBClient

client = OMDBClient()
movie = client.get_movie_by_title_with_fallback(movie_title)

if movie:
    prompt = f"""
    Create a {target_genre} trailer for:
    Title: {movie.title}
    Genre: {movie.genre}
    Plot: {movie.plot}
    Director: {movie.director}
    Starring: {movie.actors}
    """
```

### Title Cards
```python
# Generate accurate title cards
title_card = f"{movie.title} ({movie.year})"
subtitle = f"Directed by {movie.director}"
rating_text = f"IMDb: {movie.imdb_rating} | RT: {movie.get_rating_by_source('Rotten Tomatoes')}"
```

### Genre-Based Music Selection
```python
# Use official genre for better music matching
official_genres = movie.genres_list  # ['Comedy', 'Drama']
# Match with audio library based on official classification
```

## Troubleshooting

### "Movie not found"
- Check movie title spelling
- Try adding year: `client.get_movie_by_title("Dune", year=2021)`
- Search first: `results = client.search_movies("movie name")`

### Rate Limit Exceeded
- Free tier: 1,000 requests/day
- Cache is enabled by default to minimize API calls
- Consider upgrading to paid tier for higher limits

### Network Errors
- The client automatically retries up to 3 times
- Check internet connection
- Verify OMDB API is accessible: `curl http://www.omdbapi.com/?t=Matrix&apikey=83fd90a1`

## License

This utility module is part of the GenreBender project. The OMDB API is a separate service with its own [terms of use](http://www.omdbapi.com/).

## Credits

- **OMDB API**: [http://www.omdbapi.com/](http://www.omdbapi.com/)
- **Data Sources**: IMDb, Rotten Tomatoes, Metacritic
