"""
Timeline generation with LLM integration and chunked processing.
"""

import json
from typing import List, Dict
import logging
from .azure_client import AzureOpenAIClient
from .structure_prompts import StructurePrompts

logger = logging.getLogger(__name__)

class TimelineGenerator:
    """
    Generates trailer timeline structures using LLM.
    Supports chunked generation for long shot lists.
    """
    
    def __init__(self, azure_client: AzureOpenAIClient, genre: str = 'thriller'):
        """Initialize timeline generator.
        
        Args:
            azure_client: Azure OpenAI client instance.
            genre: Target genre for trailer. Defaults to 'thriller'.
        """
        self.azure_client = azure_client
        self.genre = genre
        self.prompts = StructurePrompts()
    
    def generate_timeline(self, shots: List[Dict], 
                         target_duration: int = 90) -> Dict:
        """Generate complete trailer timeline from shots.
        
        Args:
            shots: List of top-scoring shots with analysis.
            target_duration: Target trailer duration in seconds. Defaults to 90.
            
        Returns:
            Dict: Timeline dictionary with shot sequence and music cues.
        """
        # Check if we need chunked generation
        if len(shots) > 30:
            logger.info(f"Using chunked generation for {len(shots)} shots")
            return self._generate_chunked_timeline(shots, target_duration)
        else:
            logger.info(f"Generating timeline for {len(shots)} shots")
            return self._generate_full_timeline(shots, target_duration)
    
    def _generate_full_timeline(self, shots: List[Dict], 
                               target_duration: int) -> Dict:
        """Generate timeline in a single LLM call.
        
        Args:
            shots: List of shots to include in the timeline.
            target_duration: Target duration in seconds.
            
        Returns:
            Dict: Timeline dictionary containing shot sequence and music cues.
        """
        # Format prompt
        prompt_template = self.prompts.get_prompt(self.genre)
        shot_descriptions = self.prompts.format_shot_descriptions(shots)
        prompt = prompt_template.format(shot_descriptions=shot_descriptions)
        
        # Add duration constraint
        prompt += f"\n\nTarget trailer duration: {target_duration} seconds"
        
        # Add dialogue guidance if dialogue is present
        has_dialogue = any(s.get('subtitles', {}).get('has_dialogue', False) for s in shots)
        if has_dialogue:
            prompt += "\n\nNOTE: Some shots contain dialogue. Consider using actual dialogue lines for text overlays to create authentic, character-driven narrative moments."
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert trailer editor. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Generate timeline
        response = self.azure_client.generate_structured_output(
            messages=messages
        )
        
        # Parse JSON response
        try:
            timeline = json.loads(response)
            logger.info("Successfully generated timeline")
            return self._validate_timeline(timeline, shots)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse timeline JSON: {e}")
            return self._create_fallback_timeline(shots, target_duration)
    
    def _generate_chunked_timeline(self, shots: List[Dict],
                                  target_duration: int) -> Dict:
        """Generate timeline in chunks to avoid token limits.
        
        Strategy:
            1. Divide shots into intro, middle, climax sections.
            2. Generate each section with context from previous.
            3. Merge sections into complete timeline.
        
        Args:
            shots: List of shots to include in the timeline.
            target_duration: Target duration in seconds.
            
        Returns:
            Dict: Complete timeline dictionary with all sections merged.
        """
        logger.info("Starting chunked timeline generation")
        
        # Divide shots into sections
        num_shots = len(shots)
        intro_shots = shots[:num_shots//3]
        middle_shots = shots[num_shots//3:2*num_shots//3]
        climax_shots = shots[2*num_shots//3:]
        
        sections = [
            ('intro', intro_shots, 20),
            ('middle', middle_shots, 40),
            ('climax', climax_shots, 30)
        ]
        
        complete_timeline = []
        complete_music_cues = []
        context = ""
        current_timestamp = 0
        
        for section_name, section_shots, section_duration in sections:
            logger.info(f"Generating {section_name} section ({len(section_shots)} shots)")
            
            # Create section-specific prompt
            prompt = self.prompts.create_chunked_prompt(
                genre=self.genre,
                shots=section_shots,
                section=section_name,
                context=context
            )
            
            prompt += f"\n\nTarget duration for this section: ~{section_duration} seconds"
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert trailer editor. Return only valid JSON for this section."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Generate section
            response = self.azure_client.generate_structured_output(
                messages=messages,
                max_completion_tokens=2000
            )
            
            try:
                section_timeline = json.loads(response)
                
                # Adjust timestamps
                timeline_items = section_timeline.get('timeline', [])
                for item in timeline_items:
                    complete_timeline.append(item)
                
                # Adjust music cue timestamps
                music_cues = section_timeline.get('music_cues', [])
                for cue in music_cues:
                    cue['timestamp'] += current_timestamp
                    complete_music_cues.append(cue)
                
                # Update context for next section
                if timeline_items:
                    last_shots = timeline_items[-3:]
                    context = f"Previous section ended with shots: {[s['shot_id'] for s in last_shots]}"
                
                # Update timestamp
                current_timestamp += sum(item.get('duration', 0) for item in timeline_items)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse {section_name} section: {e}")
        
        complete = {
            'timeline': complete_timeline,
            'music_cues': complete_music_cues,
            'pacing_notes': f'Chunked generation for {self.genre} trailer'
        }
        
        logger.info(f"Completed chunked generation: {len(complete_timeline)} total shots")
        
        return self._validate_timeline(complete, shots)
    
    def _validate_timeline(self, timeline: Dict, shots: List[Dict]) -> Dict:
        """Validate and sanitize timeline.
        
        Filters out invalid shot IDs and calculates total duration.
        
        Args:
            timeline: Generated timeline dictionary to validate.
            shots: Available shots with valid IDs.
            
        Returns:
            Dict: Validated timeline with invalid shots removed and total
                duration calculated.
        """
        shot_ids = {s['id'] for s in shots}
        timeline_items = timeline.get('timeline', [])
        
        # Filter out invalid shot IDs
        valid_items = []
        for item in timeline_items:
            shot_id = item.get('shot_id')
            if shot_id in shot_ids:
                valid_items.append(item)
            else:
                logger.warning(f"Invalid shot_id {shot_id} in timeline, skipping")
        
        timeline['timeline'] = valid_items
        
        # Calculate total duration
        total_duration = sum(item.get('duration', 0) for item in valid_items)
        timeline['total_duration'] = total_duration
        
        logger.info(f"Timeline validated: {len(valid_items)} shots, {total_duration:.1f}s total")
        
        return timeline
    
    def _create_fallback_timeline(self, shots: List[Dict], 
                                 target_duration: int) -> Dict:
        """Create simple fallback timeline if LLM generation fails.
        
        Creates a basic timeline with progressive pacing: slower start,
        medium pace in the middle, and fast climax.
        
        Args:
            shots: List of shots to include in the timeline.
            target_duration: Target duration in seconds.
            
        Returns:
            Dict: Basic timeline dictionary with shots and default music cues.
        """
        logger.warning("Creating fallback timeline")
        
        # Use top shots in order
        timeline_items = []
        cumulative_duration = 0
        
        for shot in shots:
            if cumulative_duration >= target_duration:
                break
            
            # Determine shot duration based on position
            if cumulative_duration < target_duration * 0.3:
                duration = 2.5  # Slower start
            elif cumulative_duration < target_duration * 0.7:
                duration = 2.0  # Medium pace
            else:
                duration = 1.0  # Fast climax
            
            timeline_items.append({
                'shot_id': shot['id'],
                'duration': duration
            })
            
            cumulative_duration += duration
        
        return {
            'timeline': timeline_items,
            'music_cues': [
                {'timestamp': 0, 'type': 'intro', 'description': 'Opening music'},
                {'timestamp': target_duration * 0.6, 'type': 'climax', 'description': 'Climactic music'}
            ],
            'pacing_notes': 'Fallback timeline',
            'total_duration': cumulative_duration
        }
    
    def export_timeline(self, timeline: Dict, output_path: str) -> None:
        """Export timeline to JSON file.
        
        Args:
            timeline: Timeline dictionary to export.
            output_path: Path to save JSON file.
        """
        with open(output_path, 'w') as f:
            json.dump(timeline, f, indent=2)
        
        logger.info(f"Exported timeline to {output_path}")
