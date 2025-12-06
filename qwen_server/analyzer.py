"""
Multimodal shot analyzer using Qwen2-VL.
Processes multiple frames and audio features for comprehensive scene understanding.
"""

import base64
import io
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MultimodalAnalyzer:
    """
    Analyzes video shots using Qwen2-VL with multi-frame and audio integration.
    """
    
    def __init__(self, model, processor, config: dict):
        """
        Initialize analyzer with model and configuration.
        
        Args:
            model: Qwen2VL model instance
            processor: Qwen2VL processor
            config: Configuration dictionary
        """
        self.model = model
        self.processor = processor
        self.config = config
        
        self.enable_audio_fusion = config['processing']['enable_audio_fusion']
        self.temporal_weight = config['processing']['temporal_weight']
        self.audio_weight = config['processing']['audio_weight']
        self.max_length = config['model'].get('max_length', 512)
    
    def analyze_shot(self, shot_data: Dict) -> Dict:
        """
        Analyze a single shot with multiple frames and audio features.
        
        Args:
            shot_data: Dictionary containing:
                - images: List of base64 encoded images
                - audio_features: Audio feature dictionary (optional)
                - shot_id: Shot identifier
                - start_time, end_time, duration: Timing info
        
        Returns:
            Dictionary with caption and genre attributes
        """
        try:
            # Decode images
            images = self._decode_images(shot_data.get('images', []))
            
            if not images:
                logger.warning(f"No valid images for shot {shot_data.get('shot_id')}")
                return self._empty_analysis()
            
            # Analyze visual content
            visual_analysis = self._analyze_visual(images)
            
            # Integrate audio if available
            if self.enable_audio_fusion and shot_data.get('audio_features'):
                audio_context = self._process_audio_features(shot_data['audio_features'])
                attributes = self._fuse_multimodal(visual_analysis['attributes'], audio_context)
            else:
                attributes = visual_analysis['attributes']
            
            return {
                'caption': visual_analysis['caption'],
                'attributes': attributes
            }
            
        except Exception as e:
            logger.error(f"Error analyzing shot {shot_data.get('shot_id')}: {e}")
            return self._empty_analysis()
    
    def _decode_images(self, base64_images: List[str]) -> List[Image.Image]:
        """
        Decode base64 images to PIL Images.
        
        Args:
            base64_images: List of base64 encoded image strings
            
        Returns:
            List of PIL Images
        """
        images = []
        for img_b64 in base64_images:
            try:
                img_bytes = base64.b64decode(img_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                images.append(img)
            except Exception as e:
                logger.warning(f"Failed to decode image: {e}")
                continue
        
        return images
    
    def _analyze_visual(self, images: List[Image.Image]) -> Dict:
        """
        Analyze visual content from multiple frames.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Dictionary with caption and attributes
        """
        # For multi-frame analysis, we'll analyze key frames and aggregate
        # Using first, middle, and last frames for temporal understanding
        
        key_indices = self._select_key_frames(len(images))
        key_images = [images[i] for i in key_indices]
        
        # Analyze each key frame
        frame_analyses = []
        for img in key_images:
            analysis = self._analyze_single_frame(img)
            frame_analyses.append(analysis)
        
        # Aggregate results
        aggregated = self._aggregate_temporal(frame_analyses)
        
        return aggregated
    
    def _select_key_frames(self, num_frames: int) -> List[int]:
        """
        Select key frame indices for analysis.
        
        Args:
            num_frames: Total number of frames
            
        Returns:
            List of frame indices to analyze
        """
        if num_frames <= 3:
            return list(range(num_frames))
        
        # Use first, middle, and last frames
        return [0, num_frames // 2, num_frames - 1]
    
    def _analyze_single_frame(self, image: Image.Image) -> Dict:
        """
        Analyze a single frame using Qwen2-VL.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with caption and attributes
        """
        # Create prompt for genre attribute extraction
        prompt = """Analyze this film scene and provide:
1. A brief description of what's happening
2. Rate the following attributes from 0.0 to 1.0:
   - Suspense: Level of tension or anticipation
   - Darkness: Visual darkness and mood
   - Ambiguity: Unclear or mysterious elements
   - Emotional tension: Character stress or conflict
   - Intensity: Overall energy and impact
   - Motion: Amount of movement or action

Respond in this format:
Description: [your description]
Suspense: [0.0-1.0]
Darkness: [0.0-1.0]
Ambiguity: [0.0-1.0]
Emotional_tension: [0.0-1.0]
Intensity: [0.0-1.0]
Motion: [0.0-1.0]"""
        
        try:
            # Prepare input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Process with model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = inputs.to(self.model.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
            
            # Decode response
            generated_text = self.processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            # Parse response
            return self._parse_model_output(generated_text)
            
        except Exception as e:
            logger.error(f"Error in single frame analysis: {e}")
            return self._default_frame_analysis()
    
    def _parse_model_output(self, text: str) -> Dict:
        """
        Parse model output to extract caption and attributes.
        
        Args:
            text: Generated text from model
            
        Returns:
            Dictionary with caption and attributes
        """
        lines = text.strip().split('\n')
        caption = ""
        attributes = {}
        
        for line in lines:
            line = line.strip()
            if line.startswith("Description:"):
                caption = line.replace("Description:", "").strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                try:
                    value = float(value.strip())
                    value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
                    attributes[key] = value
                except:
                    pass
        
        # Ensure all attributes exist
        required_attrs = ['suspense', 'darkness', 'ambiguity', 
                         'emotional_tension', 'intensity', 'motion']
        for attr in required_attrs:
            if attr not in attributes:
                attributes[attr] = 0.5
        
        return {
            'caption': caption or "Scene analysis",
            'attributes': attributes
        }
    
    def _default_frame_analysis(self) -> Dict:
        """
        Return default analysis when model fails.
        
        Returns:
            Dictionary with default values
        """
        return {
            'caption': "Scene",
            'attributes': {
                'suspense': 0.5,
                'darkness': 0.5,
                'ambiguity': 0.5,
                'emotional_tension': 0.5,
                'intensity': 0.5,
                'motion': 0.5
            }
        }
    
    def _aggregate_temporal(self, frame_analyses: List[Dict]) -> Dict:
        """
        Aggregate analyses from multiple frames.
        
        Args:
            frame_analyses: List of frame analysis dictionaries
            
        Returns:
            Aggregated analysis
        """
        if not frame_analyses:
            return self._default_frame_analysis()
        
        # Combine captions
        captions = [fa['caption'] for fa in frame_analyses if fa['caption']]
        combined_caption = captions[0] if captions else "Scene"
        
        # Average attributes with temporal weighting
        # Give more weight to middle frames
        weights = self._get_temporal_weights(len(frame_analyses))
        
        aggregated_attrs = {}
        attr_keys = frame_analyses[0]['attributes'].keys()
        
        for key in attr_keys:
            values = [fa['attributes'][key] for fa in frame_analyses]
            weighted_avg = sum(v * w for v, w in zip(values, weights))
            aggregated_attrs[key] = weighted_avg
        
        # Detect motion from temporal variation
        motion_variance = np.var([fa['attributes']['intensity'] for fa in frame_analyses])
        aggregated_attrs['motion'] = min(1.0, aggregated_attrs['motion'] + motion_variance * 0.5)
        
        return {
            'caption': combined_caption,
            'attributes': aggregated_attrs
        }
    
    def _get_temporal_weights(self, num_frames: int) -> List[float]:
        """
        Get weights for temporal aggregation.
        Middle frames weighted more heavily.
        
        Args:
            num_frames: Number of frames
            
        Returns:
            List of weights (sums to 1.0)
        """
        if num_frames == 1:
            return [1.0]
        elif num_frames == 2:
            return [0.4, 0.6]
        else:  # 3 or more
            # Middle frame gets most weight
            weights = [0.25, 0.5, 0.25]
            return weights[:num_frames]
    
    def _process_audio_features(self, audio_features: Dict) -> Dict:
        """
        Process audio features to create audio context.
        
        Args:
            audio_features: Dictionary of audio features
            
        Returns:
            Audio context dictionary
        """
        if not audio_features:
            return {'type': 'unknown', 'intensity': 0.5, 'brightness': 0.5}
        
        energy = audio_features.get('rms_energy_mean', 0)
        brightness = audio_features.get('spectral_centroid_mean', 0)
        zcr = audio_features.get('zero_crossing_rate_mean', 0)
        
        # Classify audio type
        audio_type = 'silent'
        if energy > 0.05:
            if brightness > 3000:
                audio_type = 'bright'  # Music, action
            elif brightness < 1000:
                audio_type = 'dark'    # Rumble, suspense
            elif zcr > 0.15:
                audio_type = 'dialog'  # Speech
            else:
                audio_type = 'balanced'
        
        return {
            'type': audio_type,
            'intensity': min(energy * 20, 1.0),
            'brightness': brightness / 4000,
            'tempo': audio_features.get('tempo')
        }
    
    def _fuse_multimodal(self, visual_attrs: Dict, audio_context: Dict) -> Dict:
        """
        Fuse visual and audio information.
        
        Args:
            visual_attrs: Visual attribute scores
            audio_context: Audio context dictionary
            
        Returns:
            Fused attributes
        """
        fused = visual_attrs.copy()
        
        # Enhance based on audio type
        if audio_context['type'] == 'dark':
            fused['suspense'] = min(1.0, fused['suspense'] * 1.2)
            fused['darkness'] = min(1.0, fused['darkness'] * 1.15)
        elif audio_context['type'] == 'bright':
            fused['intensity'] = min(1.0, fused['intensity'] * 1.3)
        
        # Audio intensity boosts overall intensity
        fused['intensity'] = min(1.0, 
            fused['intensity'] + audio_context['intensity'] * self.audio_weight
        )
        
        return fused
    
    def _empty_analysis(self) -> Dict:
        """
        Return empty analysis for failed requests.
        
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
                'motion': 0.0
            }
        }


def process_vision_info(messages):
    """
    Process vision information from messages (helper function).
    
    Args:
        messages: Chat messages with images
        
    Returns:
        Tuple of (images, videos)
    """
    image_inputs = []
    video_inputs = []
    
    for message in messages:
        if isinstance(message.get("content"), list):
            for content in message["content"]:
                if content.get("type") == "image":
                    image_inputs.append(content["image"])
                elif content.get("type") == "video":
                    video_inputs.append(content["video"])
    
    return image_inputs, video_inputs
