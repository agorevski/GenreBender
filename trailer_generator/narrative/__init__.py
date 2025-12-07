"""
Narrative generation module for creating trailer structures with LLM.
"""

from .azure_client import AzureOpenAIClient
from .structure_prompts import StructurePrompts
from .timeline_generator import TimelineGenerator

__all__ = ['AzureOpenAIClient', 'StructurePrompts', 'TimelineGenerator']
