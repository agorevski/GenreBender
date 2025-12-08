"""
Remote analyzer client for Qwen2-VL multimodal analysis.
Communicates with external server for shot analysis.
"""

import requests
import asyncio
import aiohttp
import base64
from pathlib import Path
from typing import List, Dict, Optional
import logging
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class RemoteAnalyzer:
    """
    Client for communicating with Qwen2-VL server for multimodal shot analysis.
    Supports batch processing, retries, and network resilience.
    """
    
    def __init__(self, server_url: str, timeout: int = 30, max_retries: int = 3,
                 batch_size: int = 10, api_key: Optional[str] = None):
        """
        Initialize remote analyzer.
        
        Args:
            server_url: URL of Qwen2-VL server
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            batch_size: Number of shots to analyze per batch request
            api_key: API key for server authentication
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.api_key = api_key
        self.headers = {
            "Authorization": "Bearer " + api_key if api_key else "",
            "Content-Type": "application/json"
        }
    
    def analyze_shot(self, shot: Dict, video_path: str) -> Dict:
        """
        Analyze a single shot using remote Qwen2-VL server.
        
        Args:
            shot: Shot dictionary with keyframe paths and audio features
            video_path: Path to source video (for audio extraction)
            
        Returns:
            Analysis dictionary with caption and attributes
        """
        # Get keyframes (multiple frames or single frame)
        keyframe_paths = shot.get('keyframes', [])
        if not keyframe_paths:
            # Fallback to single keyframe for backward compatibility
            keyframe_path = shot.get('keyframe')
            if keyframe_path and Path(keyframe_path).exists():
                keyframe_paths = [keyframe_path]
        
        if not keyframe_paths:
            logger.warning(f"No valid keyframes for shot {shot.get('id')}")
            return self._empty_analysis()
        
        # Encode all keyframes as base64
        images_data = []
        for kf_path in keyframe_paths:
            if Path(kf_path).exists():
                with open(kf_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    images_data.append(image_data)
        
        if not images_data:
            logger.warning(f"Failed to encode keyframes for shot {shot.get('id')}")
            return self._empty_analysis()
        
        # Prepare request payload with multiple frames and audio features
        payload = {
            'shot_id': shot.get('id'),
            'images': images_data,  # Multiple frames
            'audio_features': shot.get('audio_features'),  # Audio features
            'start_time': shot.get('start_time'),
            'end_time': shot.get('end_time'),
            'duration': shot.get('duration')
        }
        
        # Make request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/analyze",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.debug(f"Successfully analyzed shot {shot.get('id')}")
                    return result
                else:
                    logger.warning(f"Server returned status {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for shot {shot.get('id')}")
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            
            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to analyze shot {shot.get('id')} after {self.max_retries} attempts")
        return self._empty_analysis()
    
    def analyze_batch(self, shots: List[Dict], video_path: str) -> List[Dict]:
        """
        Analyze multiple shots in a single batch request.
        
        Args:
            shots: List of shot dictionaries
            video_path: Path to source video
            
        Returns:
            List of shots with analysis results
        """
        logger.info(f"Analyzing batch of {len(shots)} shots (batch_size config: {self.batch_size})")
        
        # Prepare batch payload with multiple frames per shot
        batch_items = []
        skipped_shots = []
        
        for shot in shots:
            shot_id = shot.get('id')
            
            # Get keyframes (multiple frames or single frame)
            keyframe_paths = shot.get('keyframes', [])
            if not keyframe_paths:
                # Fallback to single keyframe for backward compatibility
                keyframe_path = shot.get('keyframe')
                if keyframe_path and Path(keyframe_path).exists():
                    keyframe_paths = [keyframe_path]
            
            if not keyframe_paths:
                logger.warning(f"Shot {shot_id}: No valid keyframe paths found")
                skipped_shots.append(shot_id)
                continue
            
            # Encode all keyframes as base64
            images_data = []
            for kf_path in keyframe_paths:
                if Path(kf_path).exists():
                    try:
                        with open(kf_path, 'rb') as f:
                            image_bytes = f.read()
                            # Validate that image file is not empty
                            if len(image_bytes) > 0:
                                image_data = base64.b64encode(image_bytes).decode('utf-8')
                                # Validate base64 encoding is not empty
                                if len(image_data) > 0:
                                    images_data.append(image_data)
                                else:
                                    logger.warning(f"Shot {shot_id}: Empty base64 encoding for {kf_path}")
                            else:
                                logger.warning(f"Shot {shot_id}: Empty image file {kf_path}")
                    except Exception as e:
                        logger.warning(f"Shot {shot_id}: Failed to encode {kf_path}: {e}")
                else:
                    logger.warning(f"Shot {shot_id}: Keyframe file not found: {kf_path}")
            
            # Only add shot if we have valid encoded images
            if images_data and len(images_data) > 0:
                batch_items.append({
                    'shot_id': shot_id,
                    'images': images_data,  # Multiple frames
                    'audio_features': shot.get('audio_features'),  # Audio features
                    'start_time': shot.get('start_time'),
                    'end_time': shot.get('end_time'),
                    'duration': shot.get('duration')
                })
            else:
                logger.warning(f"Shot {shot_id}: No valid encoded images, skipping")
                skipped_shots.append(shot_id)
        
        if skipped_shots:
            logger.warning(f"Skipped {len(skipped_shots)} shots with invalid/missing images: {skipped_shots[:10]}")
        
        if not batch_items:
            logger.error("No valid shots to analyze in batch")
            # Mark all shots as failed
            for shot in shots:
                shot['analysis'] = self._empty_analysis()
            return shots
        
        logger.info(f"Prepared batch with {len(batch_items)} valid shots (skipped {len(skipped_shots)})")
        payload = {'shots': batch_items}
        
        # Make batch request with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.server_url}/analyze_batch",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout * len(shots)  # Longer timeout for batch
                )
                
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    
                    # Match results back to shots
                    result_map = {r['shot_id']: r for r in results}
                    
                    for shot in shots:
                        shot_id = shot.get('id')
                        if shot_id in result_map:
                            shot['analysis'] = result_map[shot_id]
                        else:
                            shot['analysis'] = self._empty_analysis()
                    
                    logger.info(f"Successfully analyzed batch of {len(shots)} shots")
                    return shots
                else:
                    # Log detailed error information
                    error_detail = f"Server returned status {response.status_code}"
                    try:
                        error_body = response.json()
                        error_detail += f", detail: {error_body.get('detail', 'No detail provided')}"
                    except:
                        error_detail += f", body: {response.text[:200]}"
                    
                    logger.error(f"Batch request failed on attempt {attempt + 1}: {error_detail}")
                    
            except requests.exceptions.Timeout as e:
                logger.warning(f"Batch request timeout on attempt {attempt + 1}: {e}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Batch request connection error on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.warning(f"Batch request failed on attempt {attempt + 1}: {e}")
                
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
        
        # Fallback: analyze individually
        logger.warning("Batch request failed, falling back to individual analysis")
        for shot in shots:
            shot['analysis'] = self.analyze_shot(shot, video_path)
        
        return shots
    
    async def analyze_batch_async(self, shots: List[Dict], video_path: str) -> List[Dict]:
        """
        Analyze batch asynchronously for better performance.
        
        Args:
            shots: List of shot dictionaries
            video_path: Path to source video
            
        Returns:
            List of shots with analysis results
        """
        logger.info(f"Analyzing batch of {len(shots)} shots (async)")
        
        # Prepare batch payload with multiple frames per shot
        batch_items = []
        skipped_shots = []
        
        for shot in shots:
            shot_id = shot.get('id')
            
            # Get keyframes (multiple frames or single frame)
            keyframe_paths = shot.get('keyframes', [])
            if not keyframe_paths:
                # Fallback to single keyframe for backward compatibility
                keyframe_path = shot.get('keyframe')
                if keyframe_path and Path(keyframe_path).exists():
                    keyframe_paths = [keyframe_path]
            
            if not keyframe_paths:
                logger.warning(f"Shot {shot_id}: No valid keyframe paths found (async)")
                skipped_shots.append(shot_id)
                continue
            
            # Encode all keyframes as base64
            images_data = []
            for kf_path in keyframe_paths:
                if Path(kf_path).exists():
                    try:
                        with open(kf_path, 'rb') as f:
                            image_bytes = f.read()
                            # Validate that image file is not empty
                            if len(image_bytes) > 0:
                                image_data = base64.b64encode(image_bytes).decode('utf-8')
                                # Validate base64 encoding is not empty
                                if len(image_data) > 0:
                                    images_data.append(image_data)
                                else:
                                    logger.warning(f"Shot {shot_id}: Empty base64 encoding for {kf_path} (async)")
                            else:
                                logger.warning(f"Shot {shot_id}: Empty image file {kf_path} (async)")
                    except Exception as e:
                        logger.warning(f"Shot {shot_id}: Failed to encode {kf_path}: {e} (async)")
                else:
                    logger.warning(f"Shot {shot_id}: Keyframe file not found: {kf_path} (async)")
            
            # Only add shot if we have valid encoded images
            if images_data and len(images_data) > 0:
                batch_items.append({
                    'shot_id': shot_id,
                    'images': images_data,  # Multiple frames
                    'audio_features': shot.get('audio_features'),  # Audio features
                    'start_time': shot.get('start_time'),
                    'end_time': shot.get('end_time'),
                    'duration': shot.get('duration')
                })
            else:
                logger.warning(f"Shot {shot_id}: No valid encoded images, skipping (async)")
                skipped_shots.append(shot_id)
        
        if skipped_shots:
            logger.warning(f"Skipped {len(skipped_shots)} shots (async): {skipped_shots[:10]}")
        
        if not batch_items:
            logger.error("No valid shots to analyze in batch (async)")
            # Mark all shots as failed
            for shot in shots:
                shot['analysis'] = self._empty_analysis()
            return shots
        
        logger.info(f"Prepared async batch with {len(batch_items)} valid shots (skipped {len(skipped_shots)})")
        payload = {'shots': batch_items}
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        f"{self.server_url}/analyze_batch",
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout * len(shots))
                    ) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            results = data.get('results', [])
                            
                            # Match results back to shots
                            result_map = {r['shot_id']: r for r in results}
                            
                            for shot in shots:
                                shot_id = shot.get('id')
                                if shot_id in result_map:
                                    shot['analysis'] = result_map[shot_id]
                                else:
                                    shot['analysis'] = self._empty_analysis()
                            
                            logger.info(f"Successfully analyzed batch (async)")
                            return shots
                            
                except Exception as e:
                    logger.warning(f"Async batch failed on attempt {attempt + 1}: {e}")
                    
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
        
        # Fallback
        logger.warning("Async batch failed, using synchronous fallback")
        return self.analyze_batch(shots, video_path)
    
    def _empty_analysis(self) -> Dict:
        """
        Return empty analysis structure for failed requests.
        
        Returns:
            Dictionary with default values
        """
        return {
            'caption': 'Analysis failed',
            'attributes': {
                'suspense': 0.0,
                'darkness': 0.0,
                'ambiguity': 0.0,
                'emotional_tension': 0.0,
                'intensity': 0.0,
                'motion': 0.0,
                'impact': 0.0,
                'energy': 0.0,
                'emotional_connection': 0.0,
                'intimacy': 0.0,
                'warmth': 0.0,
                'fear': 0.0,
                'unease': 0.0,
                'shock': 0.0,
                'futuristic': 0.0,
                'technology': 0.0,
                'wonder': 0.0,
                'scale': 0.0,
                'humor': 0.0,
                'lightheartedness': 0.0,
                'timing': 0.0,
                'beauty': 0.0
            }
        }
    
    def health_check(self) -> bool:
        """
        Check if remote server is available.
        
        Returns:
            True if server is responding
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                headers=self.headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class MultiServerAnalyzer:
    """
    Multi-server analyzer that distributes workload across multiple Qwen2-VL servers.
    Enables true multi-GPU parallelism with one server per GPU.
    """
    
    def __init__(self, server_urls: List[str], load_balancing: str = "round_robin",
                 timeout: int = 30, max_retries: int = 3, batch_size: int = 10,
                 api_key: Optional[str] = None):
        """
        Initialize multi-server analyzer.
        
        Args:
            server_urls: List of server URLs (one per GPU)
            load_balancing: Load balancing strategy ('round_robin' or 'random')
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            batch_size: Number of shots per batch request
            api_key: API key for server authentication
        """
        self.server_urls = server_urls
        self.load_balancing = load_balancing
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.api_key = api_key
        
        # Create analyzer for each server
        self.analyzers = [
            RemoteAnalyzer(url, timeout, max_retries, batch_size, api_key)
            for url in server_urls
        ]
        
        # Round-robin counter
        self.current_idx = 0
        
        logger.info(f"Initialized MultiServerAnalyzer with {len(server_urls)} servers")
        logger.info(f"Load balancing strategy: {load_balancing}")
    
    def _select_server(self) -> RemoteAnalyzer:
        """
        Select next server based on load balancing strategy.
        
        Returns:
            RemoteAnalyzer instance
        """
        if self.load_balancing == "round_robin":
            analyzer = self.analyzers[self.current_idx]
            self.current_idx = (self.current_idx + 1) % len(self.analyzers)
            return analyzer
        elif self.load_balancing == "random":
            return random.choice(self.analyzers)
        else:
            # Default to round-robin
            return self._select_server()
    
    def _split_shots(self, shots: List[Dict]) -> List[List[Dict]]:
        """
        Split shots into N chunks (one per server).
        
        Args:
            shots: List of shot dictionaries
            
        Returns:
            List of shot chunks
        """
        num_servers = len(self.analyzers)
        chunk_size = len(shots) // num_servers
        remainder = len(shots) % num_servers
        
        chunks = []
        start = 0
        
        for i in range(num_servers):
            # Distribute remainder across first few chunks
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append(shots[start:start + size])
            start += size
        
        return chunks
    
    def analyze_batch(self, shots: List[Dict], video_path: str) -> List[Dict]:
        """
        Analyze shots by distributing across all servers in parallel.
        
        Args:
            shots: List of shot dictionaries
            video_path: Path to source video
            
        Returns:
            List of shots with analysis results
        """
        num_shots = len(shots)
        num_servers = len(self.analyzers)
        
        logger.info(f"Multi-server batch analysis: {num_shots} shots across {num_servers} servers")
        
        # Split shots into chunks
        chunks = self._split_shots(shots)
        
        # Log distribution
        for i, chunk in enumerate(chunks):
            logger.info(f"  Server {i+1} ({self.server_urls[i]}): {len(chunk)} shots")
        
        # Analyze each chunk on different server in parallel
        results = []
        
        with ThreadPoolExecutor(max_workers=num_servers) as executor:
            # Submit all chunks
            future_to_chunk = {}
            for analyzer, chunk in zip(self.analyzers, chunks):
                if chunk:  # Only submit if chunk has shots
                    future = executor.submit(analyzer.analyze_batch, chunk, video_path)
                    future_to_chunk[future] = chunk
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    analyzed_chunk = future.result()
                    results.extend(analyzed_chunk)
                    logger.info(f"Completed analyzing chunk of {len(chunk)} shots")
                except Exception as e:
                    logger.error(f"Error analyzing chunk: {e}")
                    # Add failed shots with empty analysis
                    for shot in chunk:
                        shot['analysis'] = self._empty_analysis()
                        results.append(shot)
        
        logger.info(f"Multi-server batch complete: {len(results)} shots analyzed")
        
        return results
    
    def health_check(self) -> bool:
        """
        Check if all servers are available.
        
        Returns:
            True if all servers are responding
        """
        logger.info(f"Health checking {len(self.analyzers)} servers...")
        
        all_healthy = True
        for i, analyzer in enumerate(self.analyzers):
            url = self.server_urls[i]
            if analyzer.health_check():
                logger.info(f"  ✓ Server {i+1} ({url}): Healthy")
            else:
                logger.error(f"  ✗ Server {i+1} ({url}): Not responding")
                all_healthy = False
        
        return all_healthy
    
    def _empty_analysis(self) -> Dict:
        """
        Return empty analysis structure for failed requests.
        
        Returns:
            Dictionary with default values
        """
        return {
            'caption': 'Analysis failed',
            'attributes': {
                'suspense': 0.0,
                'darkness': 0.0,
                'ambiguity': 0.0,
                'emotional_tension': 0.0,
                'intensity': 0.0,
                'motion': 0.0,
                'impact': 0.0,
                'energy': 0.0,
                'emotional_connection': 0.0,
                'intimacy': 0.0,
                'warmth': 0.0,
                'fear': 0.0,
                'unease': 0.0,
                'shock': 0.0,
                'futuristic': 0.0,
                'technology': 0.0,
                'wonder': 0.0,
                'scale': 0.0,
                'humor': 0.0,
                'lightheartedness': 0.0,
                'timing': 0.0,
                'beauty': 0.0
            }
        }
