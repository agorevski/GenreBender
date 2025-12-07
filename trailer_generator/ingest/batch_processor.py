"""
Batch processing utilities for handling large video files efficiently.
Prevents memory overload when processing full-length movies.
"""

from typing import List, Dict, Iterator, Callable, Any
import logging

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Handles batch processing of shots to avoid memory issues.
    Yields batches for streaming processing pipelines.
    """
    
    def __init__(self, batch_size: int = 10):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of items to process per batch
        """
        self.batch_size = batch_size
    
    def batch_shots(self, shots: List[Dict]) -> Iterator[List[Dict]]:
        """
        Yield shots in batches.
        
        Args:
            shots: List of all shots
            
        Yields:
            Batches of shots
        """
        total_shots = len(shots)
        logger.info(f"Processing {total_shots} shots in batches of {self.batch_size}")
        
        for i in range(0, total_shots, self.batch_size):
            batch = shots[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_shots + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Yielding batch {batch_num}/{total_batches} ({len(batch)} shots)")
            yield batch
    
    def process_batches(self, shots: List[Dict], 
                       processor_func: Callable[[List[Dict]], List[Dict]],
                       **kwargs) -> List[Dict]:
        """
        Process shots in batches using provided processor function.
        
        Args:
            shots: List of shots to process
            processor_func: Function to apply to each batch
            **kwargs: Additional arguments to pass to processor_func
            
        Returns:
            List of all processed shots
        """
        all_processed = []
        
        for batch in self.batch_shots(shots):
            try:
                processed_batch = processor_func(batch, **kwargs)
                all_processed.extend(processed_batch)
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Continue with next batch
                all_processed.extend(batch)  # Add unprocessed batch
        
        return all_processed
    
    def process_batches_async(self, shots: List[Dict],
                             async_processor: Callable) -> List[Dict]:
        """
        Process batches asynchronously for remote API calls.
        Useful for Qwen2-VL server communication.
        
        Args:
            shots: List of shots to process
            async_processor: Async function to process batches
            
        Returns:
            List of all processed shots
        """
        import asyncio
        
        async def _process_all():
            all_processed = []
            tasks = []
            
            for batch in self.batch_shots(shots):
                task = asyncio.create_task(async_processor(batch))
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing failed: {result}")
                else:
                    all_processed.extend(result)
            
            return all_processed
        
        return asyncio.run(_process_all())
    
    def save_partial_results(self, shots: List[Dict], filepath: str):
        """
        Save partial results to allow pipeline resumption.
        
        Args:
            shots: Shots to save
            filepath: Path to save JSON
        """
        import json
        from pathlib import Path
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({'shots': shots}, f, indent=2)
        
        logger.info(f"Saved {len(shots)} partial results to {filepath}")
    
    def load_partial_results(self, filepath: str) -> List[Dict]:
        """
        Load partial results to resume processing.
        
        Args:
            filepath: Path to saved JSON
            
        Returns:
            List of shots or empty list if file doesn't exist
        """
        import json
        from pathlib import Path
        
        output_path = Path(filepath)
        
        if not output_path.exists():
            logger.info("No partial results found")
            return []
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        shots = data.get('shots', [])
        logger.info(f"Loaded {len(shots)} partial results from {filepath}")
        
        return shots
