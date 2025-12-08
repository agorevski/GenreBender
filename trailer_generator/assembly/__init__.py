"""
Video assembly module for trailer generation.
Handles video concatenation, color grading, and transitions.
"""

from .video_assembler import VideoAssembler
from .title_generator import TitleGenerator
from .transition_selector import TransitionSelector

__all__ = ['VideoAssembler', 'TitleGenerator', 'TransitionSelector']
