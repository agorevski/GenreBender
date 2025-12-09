"""
Azure OpenAI client for GPT-4 API integration.
Handles authentication, rate limiting, and retries.
"""

from openai import AzureOpenAI
from typing import List, Dict, Optional
import time
import logging
import os

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """
    Client for Azure OpenAI API with retry logic and rate limiting.
    """
    def __init__(self, endpoint: str, api_key: str, deployment_name: str, api_version: str = "2025-01-01-preview", max_retries: int = 3, temperature: float = 0.7, max_completion_tokens: int = 50000):
        """
        Initialize Azure OpenAI client.
        
        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: API key for authentication
            deployment_name: Name of deployed model
            api_version: API version to use
            max_retries: Maximum retry attempts
            temperature: Default sampling temperature
            max_completion_tokens: Default maximum tokens for completion
        """
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv('AZURE_OPENAI_KEY')
        self.deployment_name = deployment_name
        self.api_version = api_version
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        
        if not self.api_key:
            raise ValueError("Azure OpenAI API key not provided")
        
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"Initialized Azure OpenAI client with deployment: {deployment_name}")
    
    def generate_completion(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_completion_tokens: Optional[int] = None, response_format: Optional[Dict] = None) -> str:
        """
        Generate completion from Azure OpenAI.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (uses default from config if not specified)
            max_completion_tokens: Maximum tokens in response (uses default from config if not specified)
            response_format: Optional format specification (e.g., {"type": "json_object"})
            
        Returns:
            Generated completion text
        """
        # Use instance defaults if not specified
        if temperature is None:
            temperature = self.temperature
        if max_completion_tokens is None:
            max_completion_tokens = self.max_completion_tokens
        
        # Log request details
        prompt_chars = sum(len(m.get('content', '')) for m in messages)
        estimated_prompt_tokens = self.estimate_tokens(str(prompt_chars))
        logger.info(f"API Request: {len(messages)} messages, ~{prompt_chars} chars, "
                   f"~{estimated_prompt_tokens} tokens, temp={temperature}, "
                   f"max_tokens={max_completion_tokens}")
        logger.debug(f"Request format: {response_format}")
            
        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.deployment_name,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": max_completion_tokens
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                
                # Enhanced response logging
                choice = response.choices[0]
                completion = choice.message.content or ""
                finish_reason = choice.finish_reason
                
                # Log usage stats
                if hasattr(response, 'usage') and response.usage:
                    logger.info(f"API Response: {len(completion)} chars, "
                               f"finish_reason={finish_reason}, "
                               f"tokens: prompt={response.usage.prompt_tokens}, "
                               f"completion={response.usage.completion_tokens}, "
                               f"total={response.usage.total_tokens}")
                else:
                    logger.info(f"API Response: {len(completion)} chars, "
                               f"finish_reason={finish_reason}")
                
                # Check for content filtering
                if hasattr(choice, 'content_filter_results') and choice.content_filter_results:
                    logger.warning(f"Content filter results: {choice.content_filter_results}")
                
                # Handle truncated responses (finish_reason=length)
                if finish_reason == 'length':
                    logger.warning(f"Response truncated due to token limit! Received {len(completion)} chars")
                    if completion:
                        logger.warning(f"Response preview: {completion[:200]}...")
                        logger.warning(f"Response end: ...{completion[-200:]}")
                
                # Log warning for empty responses
                if not completion:
                    logger.error(f"Empty completion received! finish_reason={finish_reason}")
                    if hasattr(choice, 'content_filter_results'):
                        logger.error(f"Content filter results: {choice.content_filter_results}")
                    logger.debug(f"Full response object: {response}")
                else:
                    logger.info(f"Generated completion ({len(completion)} chars)")
                
                return completion
                
            except Exception as e:
                logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts")
                    raise

    def generate_structured_output(self, messages: List[Dict[str, str]], temperature: Optional[float] = None, max_completion_tokens: Optional[int] = None) -> str:
        """
        Generate structured JSON output.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (uses default from config if not specified)
            max_completion_tokens: Maximum tokens in response (uses default from config if not specified)
            
        Returns:
            JSON string
        """
        return self.generate_completion(
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"}
        )
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Rough approximation: ~4 chars per token
        return len(text) // 4
    
    def check_token_limit(self, messages: List[Dict[str, str]], max_tokens: int = 50000) -> bool:
        """
        Check if messages fit within token limit.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum allowed tokens
            
        Returns:
            True if within limit
        """
        total_chars = sum(len(m.get('content', '')) for m in messages)
        estimated_tokens = self.estimate_tokens(str(total_chars))
        
        within_limit = estimated_tokens < max_tokens
        
        if not within_limit:
            logger.warning(f"Estimated {estimated_tokens} tokens exceeds limit of {max_tokens}")
        
        return within_limit
    
    def truncate_messages(self, messages: List[Dict[str, str]], max_tokens: int = 8000) -> List[Dict[str, str]]:
        """
        Truncate messages to fit token limit.
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum allowed tokens
            
        Returns:
            Truncated messages list
        """
        if self.check_token_limit(messages, max_tokens):
            return messages
        
        # Keep system message and truncate user content
        truncated = []
        
        for msg in messages:
            if msg['role'] == 'system':
                truncated.append(msg)
            else:
                content = msg['content']
                estimated_tokens = self.estimate_tokens(content)
                
                if estimated_tokens > max_tokens // 2:
                    # Truncate content
                    target_chars = (max_tokens // 2) * 4
                    truncated_content = content[:target_chars] + "...[truncated]"
                    truncated.append({
                        'role': msg['role'],
                        'content': truncated_content
                    })
                else:
                    truncated.append(msg)
        
        logger.info("Truncated messages to fit token limit")
        
        return truncated
