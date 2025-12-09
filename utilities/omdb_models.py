"""
OMDB API Data Models
====================

Data classes for OMDB API responses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class OMDBRating:
    """Represents a rating from a specific source (IMDb, Rotten Tomatoes, etc.)"""
    source: str
    value: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {
            'Source': self.source,
            'Value': self.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'OMDBRating':
        """Create from dictionary"""
        return cls(
            source=data.get('Source', ''),
            value=data.get('Value', '')
        )


@dataclass
class OMDBMovie:
    """
    Represents a movie from the OMDB API.
    
    Attributes:
        title: Movie title
        year: Release year
        rated: Content rating (e.g., PG-13, R)
        released: Release date
        runtime: Duration (e.g., "107 min")
        genre: Comma-separated genres
        director: Director name(s)
        writer: Writer name(s)
        actors: Comma-separated actor names
        plot: Full plot description
        language: Comma-separated languages
        country: Country of origin
        awards: Awards and nominations
        poster_url: URL to movie poster image
        ratings: List of ratings from various sources
        metascore: Metascore rating
        imdb_rating: IMDb rating
        imdb_votes: Number of IMDb votes
        imdb_id: IMDb ID (e.g., tt0109686)
        type: Media type (movie, series, episode)
        dvd: DVD release date
        box_office: Box office earnings
        production: Production company
        website: Official website
        response: API response status
    """
    title: str
    year: str
    rated: str
    released: str
    runtime: str
    genre: str
    director: str
    writer: str
    actors: str
    plot: str
    language: str
    country: str
    awards: str
    poster_url: str
    ratings: List[OMDBRating] = field(default_factory=list)
    metascore: str = "N/A"
    imdb_rating: str = "N/A"
    imdb_votes: str = "N/A"
    imdb_id: str = ""
    type: str = "movie"
    dvd: str = "N/A"
    box_office: str = "N/A"
    production: str = "N/A"
    website: str = "N/A"
    response: str = "True"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching OMDB API format"""
        return {
            'Title': self.title,
            'Year': self.year,
            'Rated': self.rated,
            'Released': self.released,
            'Runtime': self.runtime,
            'Genre': self.genre,
            'Director': self.director,
            'Writer': self.writer,
            'Actors': self.actors,
            'Plot': self.plot,
            'Language': self.language,
            'Country': self.country,
            'Awards': self.awards,
            'Poster': self.poster_url,
            'Ratings': [r.to_dict() for r in self.ratings],
            'Metascore': self.metascore,
            'imdbRating': self.imdb_rating,
            'imdbVotes': self.imdb_votes,
            'imdbID': self.imdb_id,
            'Type': self.type,
            'DVD': self.dvd,
            'BoxOffice': self.box_office,
            'Production': self.production,
            'Website': self.website,
            'Response': self.response
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OMDBMovie':
        """
        Create OMDBMovie from OMDB API response dictionary.
        
        Args:
            data: Dictionary from OMDB API response
            
        Returns:
            OMDBMovie instance
        """
        # Parse ratings
        ratings = []
        if 'Ratings' in data and isinstance(data['Ratings'], list):
            ratings = [OMDBRating.from_dict(r) for r in data['Ratings']]
        
        return cls(
            title=data.get('Title', ''),
            year=data.get('Year', ''),
            rated=data.get('Rated', 'N/A'),
            released=data.get('Released', 'N/A'),
            runtime=data.get('Runtime', 'N/A'),
            genre=data.get('Genre', ''),
            director=data.get('Director', ''),
            writer=data.get('Writer', ''),
            actors=data.get('Actors', ''),
            plot=data.get('Plot', ''),
            language=data.get('Language', ''),
            country=data.get('Country', ''),
            awards=data.get('Awards', ''),
            poster_url=data.get('Poster', ''),
            ratings=ratings,
            metascore=data.get('Metascore', 'N/A'),
            imdb_rating=data.get('imdbRating', 'N/A'),
            imdb_votes=data.get('imdbVotes', 'N/A'),
            imdb_id=data.get('imdbID', ''),
            type=data.get('Type', 'movie'),
            dvd=data.get('DVD', 'N/A'),
            box_office=data.get('BoxOffice', 'N/A'),
            production=data.get('Production', 'N/A'),
            website=data.get('Website', 'N/A'),
            response=data.get('Response', 'True')
        )
    
    @property
    def genres_list(self) -> List[str]:
        """Get genres as a list"""
        return [g.strip() for g in self.genre.split(',') if g.strip()]
    
    @property
    def actors_list(self) -> List[str]:
        """Get actors as a list"""
        return [a.strip() for a in self.actors.split(',') if a.strip()]
    
    @property
    def languages_list(self) -> List[str]:
        """Get languages as a list"""
        return [l.strip() for l in self.language.split(',') if l.strip()]
    
    @property
    def runtime_minutes(self) -> Optional[int]:
        """Extract runtime in minutes as integer"""
        try:
            return int(self.runtime.split()[0])
        except (ValueError, IndexError):
            return None
    
    @property
    def imdb_rating_float(self) -> Optional[float]:
        """Get IMDb rating as float"""
        try:
            return float(self.imdb_rating)
        except ValueError:
            return None
    
    @property
    def metascore_int(self) -> Optional[int]:
        """Get Metascore as integer"""
        try:
            return int(self.metascore)
        except ValueError:
            return None
    
    def get_rating_by_source(self, source: str) -> Optional[str]:
        """
        Get rating value for a specific source.
        
        Args:
            source: Source name (e.g., "Rotten Tomatoes", "Metacritic")
            
        Returns:
            Rating value or None if not found
        """
        for rating in self.ratings:
            if rating.source.lower() == source.lower():
                return rating.value
        return None
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.title} ({self.year}) - {self.genre} - IMDb: {self.imdb_rating}"
