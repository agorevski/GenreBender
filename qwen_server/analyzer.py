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
        
        # Video processing configuration
        self.use_native_video = config['processing'].get('use_native_video', False)
        self.video_max_frames = config['processing'].get('video_max_frames', 16)
        self.video_fps_sampling = config['processing'].get('video_fps_sampling', 1.0)
        self.use_parallel_batching = config['processing'].get('use_parallel_batching', False)
    
    def analyze_shot(self, shot_data: Dict) -> Dict:
        """
        Analyze a single shot with multiple frames and audio features.
        
        Args:
            shot_data: Dictionary containing:
                - images: List of base64 encoded images (for keyframe mode)
                - video: Base64 encoded video file (for video mode)
                - audio_features: Audio feature dictionary (optional)
                - shot_id: Shot identifier
                - start_time, end_time, duration: Timing info
        
        Returns:
            Dictionary with caption and genre attributes
        """
        try:
            # Check if video mode and video data provided
            if self.use_native_video and shot_data.get('video'):
                visual_analysis = self._analyze_video(shot_data)
            else:
                # Decode images for keyframe mode
                images = self._decode_images(shot_data.get('images', []))
                
                if not images:
                    logger.warning(f"No valid images for shot {shot_data.get('shot_id')}")
                    return self._empty_analysis()
                
                # Analyze visual content from keyframes
                visual_analysis = self._analyze_visual(images)
            
            # Integrate audio if available
            if self.enable_audio_fusion and shot_data.get('audio_features'):
                audio_context = self._process_audio_features(shot_data['audio_features'])
                attributes = self._fuse_multimodal(visual_analysis['attributes'], audio_context)
            else:
                attributes = visual_analysis['attributes']
            
            return {
                'caption': visual_analysis['caption'],
                'attributes': attributes,
                'processing_mode': 'video' if self.use_native_video and shot_data.get('video') else 'keyframes'
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
    
    def _analyze_video(self, shot_data: Dict) -> Dict:
        """
        Analyze a video clip using Qwen2-VL native video processing.
        
        Args:
            shot_data: Dictionary containing video data and metadata
            
        Returns:
            Dictionary with caption and attributes
        """
        try:
            # Decode base64 video to bytes
            video_b64 = shot_data.get('video', '')
            video_bytes = base64.b64decode(video_b64)
            
            # Save temporarily to process (Qwen2-VL expects file path or PIL frames)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                tmp_video.write(video_bytes)
                video_path = tmp_video.name
            
            try:
                # Create prompt for genre attribute extraction
                prompt = """Analyze this film scene and provide:
1. A brief description of what's happening across the entire clip
2. Rate the following attributes from 0.0 to 1.0:
   - Suspense: Level of tension or anticipation
   - Darkness: Visual darkness and mood
   - Ambiguity: Unclear or mysterious elements
   - Emotional_tension: Character stress or conflict
   - Intensity: Overall energy and impact
   - Motion: Amount of movement or action
   - Impact: Visual or narrative impact
   - Energy: Overall vitality and dynamism
   - Emotional_connection: Relatability and emotional resonance
   - Intimacy: Closeness and personal connection
   - Warmth: Comfort and positive emotional tone
   - Fear: Frightening or disturbing elements
   - Unease: Discomfort or unsettling atmosphere
   - Shock: Surprising or jarring moments
   - Futuristic: Science fiction or advanced technology themes
   - Technology: Presence of technological elements
   - Wonder: Sense of awe or amazement
   - Scale: Epic scope or grandeur
   - Humor: Comedic or amusing elements
   - Lightheartedness: Playful and carefree mood
   - Timing: Comedic or dramatic timing quality
   - Beauty: Aesthetic beauty or visual appeal

Respond in this format:
Description: [your description]
Suspense: [0.0-1.0]
Darkness: [0.0-1.0]
Ambiguity: [0.0-1.0]
Emotional_tension: [0.0-1.0]
Intensity: [0.0-1.0]
Motion: [0.0-1.0]
Impact: [0.0-1.0]
Energy: [0.0-1.0]
Emotional_connection: [0.0-1.0]
Intimacy: [0.0-1.0]
Warmth: [0.0-1.0]
Fear: [0.0-1.0]
Unease: [0.0-1.0]
Shock: [0.0-1.0]
Futuristic: [0.0-1.0]
Technology: [0.0-1.0]
Wonder: [0.0-1.0]
Scale: [0.0-1.0]
Humor: [0.0-1.0]
Lightheartedness: [0.0-1.0]
Timing: [0.0-1.0]
Beauty: [0.0-1.0]"""
                
                # Calculate optimal FPS to stay within frame limit
                duration = shot_data.get('duration', 10.0)
                fps = self.video_fps_sampling if self.video_fps_sampling else 1.0
                estimated_frames = int(duration * fps)
                
                # Adjust FPS if we'd exceed max frames
                if estimated_frames > self.video_max_frames:
                    fps = self.video_max_frames / duration
                    logger.info(f"Adjusting FPS from {self.video_fps_sampling} to {fps:.2f} to stay within {self.video_max_frames} frame limit")
                
                # Prepare input with video
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": 360 * 420,  # Qwen2-VL video resolution limit
                                "fps": fps,
                                "nframes": self.video_max_frames  # Explicitly limit frames
                            },
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
                    images=None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = inputs.to(self.model.device)
                
                # Generate with mixed precision for better GPU utilization
                logger.info(f"Processing video clip (duration: {shot_data.get('duration', 0):.2f}s) with native video mode")
                with torch.no_grad():
                    # Use automatic mixed precision if CUDA is available
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=256,
                                do_sample=False
                            )
                    else:
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                
                # Decode response
                output_ids = outputs[:, inputs.input_ids.shape[1]:]
                generated_texts = self.processor.batch_decode(
                    output_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                
                if not generated_texts or len(generated_texts) == 0:
                    logger.warning("Video analysis: batch_decode returned empty, using default")
                    return self._default_frame_analysis()
                
                # Parse response
                result = self._parse_model_output(generated_texts[0])
                logger.info(f"Video analysis completed: {result['caption'][:50]}...")
                return result
                
            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(video_path)
                except:
                    pass
                    
        except Exception as e:
            import traceback
            logger.error(f"Error in video analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._default_frame_analysis()
    
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
   - Emotional_tension: Character stress or conflict
   - Intensity: Overall energy and impact
   - Motion: Amount of movement or action
   - Impact: Visual or narrative impact
   - Energy: Overall vitality and dynamism
   - Emotional_connection: Relatability and emotional resonance
   - Intimacy: Closeness and personal connection
   - Warmth: Comfort and positive emotional tone
   - Fear: Frightening or disturbing elements
   - Unease: Discomfort or unsettling atmosphere
   - Shock: Surprising or jarring moments
   - Futuristic: Science fiction or advanced technology themes
   - Technology: Presence of technological elements
   - Wonder: Sense of awe or amazement
   - Scale: Epic scope or grandeur
   - Humor: Comedic or amusing elements
   - Lightheartedness: Playful and carefree mood
   - Timing: Comedic or dramatic timing quality
   - Beauty: Aesthetic beauty or visual appeal

Respond in this format:
Description: [your description]
Suspense: [0.0-1.0]
Darkness: [0.0-1.0]
Ambiguity: [0.0-1.0]
Emotional_tension: [0.0-1.0]
Intensity: [0.0-1.0]
Motion: [0.0-1.0]
Impact: [0.0-1.0]
Energy: [0.0-1.0]
Emotional_connection: [0.0-1.0]
Intimacy: [0.0-1.0]
Warmth: [0.0-1.0]
Fear: [0.0-1.0]
Unease: [0.0-1.0]
Shock: [0.0-1.0]
Futuristic: [0.0-1.0]
Technology: [0.0-1.0]
Wonder: [0.0-1.0]
Scale: [0.0-1.0]
Humor: [0.0-1.0]
Lightheartedness: [0.0-1.0]
Timing: [0.0-1.0]
Beauty: [0.0-1.0]"""
        
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
                videos=None,  # We only process images, not videos
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device (handle DataParallel wrapper)
            if isinstance(self.model, torch.nn.DataParallel):
                device = self.model.module.device
            else:
                device = self.model.device
            inputs = inputs.to(device)
            
            # Generate with mixed precision for better GPU utilization
            # Handle DataParallel wrapper - need to access .module.generate()
            model_to_use = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            with torch.no_grad():
                # Use automatic mixed precision if CUDA is available
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = model_to_use.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                else:
                    outputs = model_to_use.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )
            
            # Decode response
            output_ids = outputs[:, inputs.input_ids.shape[1]:]
            
            # Add debug logging
            logger.debug(f"Output shape: {outputs.shape}, Input shape: {inputs.input_ids.shape}")
            logger.debug(f"Output IDs shape: {output_ids.shape}")
            
            generated_texts = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Check if decoding produced any results
            if not generated_texts or len(generated_texts) == 0:
                logger.warning("batch_decode returned empty list, using default analysis")
                return self._default_frame_analysis()
            
            # Get first result (should only be one in batch)
            generated_text = generated_texts[0]
            
            # Parse response
            return self._parse_model_output(generated_text)
            
        except Exception as e:
            import traceback
            logger.error(f"Error in single frame analysis: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        
        # Ensure all attributes exist with default values
        required_attrs = [
            'suspense', 'darkness', 'ambiguity', 'emotional_tension', 
            'intensity', 'motion', 'impact', 'energy', 'emotional_connection',
            'intimacy', 'warmth', 'fear', 'unease', 'shock', 'futuristic',
            'technology', 'wonder', 'scale', 'humor', 'lightheartedness',
            'timing', 'beauty'
        ]
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
                'motion': 0.5,
                'impact': 0.5,
                'energy': 0.5,
                'emotional_connection': 0.5,
                'intimacy': 0.5,
                'warmth': 0.5,
                'fear': 0.5,
                'unease': 0.5,
                'shock': 0.5,
                'futuristic': 0.5,
                'technology': 0.5,
                'wonder': 0.5,
                'scale': 0.5,
                'humor': 0.5,
                'lightheartedness': 0.5,
                'timing': 0.5,
                'beauty': 0.5
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
    
    def analyze_batch_parallel(self, shots_data: List[Dict]) -> List[Dict]:
        """
        Analyze multiple shots in parallel using true batched inference.
        All shots are processed in a single model forward pass for maximum GPU utilization.
        
        Args:
            shots_data: List of shot dictionaries, each containing images/video and metadata
            
        Returns:
            List of analysis results, one per shot
        """
        if not self.use_parallel_batching:
            # Fallback to sequential processing
            logger.info("Parallel batching disabled, using sequential processing")
            return [self.analyze_shot(shot) for shot in shots_data]
        
        logger.info(f"Processing {len(shots_data)} shots with parallel batching")
        
        try:
            # Use keyframe mode for parallel batching (video mode is sequential)
            if self.use_native_video:
                logger.warning("Parallel batching not supported with native video mode, falling back to sequential")
                return [self.analyze_shot(shot) for shot in shots_data]
            
            # Decode all images from all shots
            all_images = []
            shot_image_counts = []
            
            for shot in shots_data:
                images = self._decode_images(shot.get('images', []))
                if not images:
                    logger.warning(f"No valid images for shot {shot.get('shot_id')}, using empty")
                    images = []
                all_images.extend(images)
                shot_image_counts.append(len(images))
            
            if not all_images:
                logger.error("No valid images in entire batch")
                return [self._empty_analysis() for _ in shots_data]
            
            # Select key frames for each shot
            shot_key_images = []
            image_idx = 0
            for count in shot_image_counts:
                if count == 0:
                    shot_key_images.append([])
                    continue
                    
                shot_images = all_images[image_idx:image_idx + count]
                key_indices = self._select_key_frames(count)
                key_images = [shot_images[i] for i in key_indices]
                shot_key_images.append(key_images)
                image_idx += count
            
            # Create batch prompt (same for all shots)
            prompt = """Analyze this film scene and provide:
1. A brief description of what's happening
2. Rate the following attributes from 0.0 to 1.0:
   - Suspense: Level of tension or anticipation
   - Darkness: Visual darkness and mood
   - Ambiguity: Unclear or mysterious elements
   - Emotional_tension: Character stress or conflict
   - Intensity: Overall energy and impact
   - Motion: Amount of movement or action
   - Impact: Visual or narrative impact
   - Energy: Overall vitality and dynamism
   - Emotional_connection: Relatability and emotional resonance
   - Intimacy: Closeness and personal connection
   - Warmth: Comfort and positive emotional tone
   - Fear: Frightening or disturbing elements
   - Unease: Discomfort or unsettling atmosphere
   - Shock: Surprising or jarring moments
   - Futuristic: Science fiction or advanced technology themes
   - Technology: Presence of technological elements
   - Wonder: Sense of awe or amazement
   - Scale: Epic scope or grandeur
   - Humor: Comedic or amusing elements
   - Lightheartedness: Playful and carefree mood
   - Timing: Comedic or dramatic timing quality
   - Beauty: Aesthetic beauty or visual appeal

Respond in this format:
Description: [your description]
Suspense: [0.0-1.0]
Darkness: [0.0-1.0]
Ambiguity: [0.0-1.0]
Emotional_tension: [0.0-1.0]
Intensity: [0.0-1.0]
Motion: [0.0-1.0]
Impact: [0.0-1.0]
Energy: [0.0-1.0]
Emotional_connection: [0.0-1.0]
Intimacy: [0.0-1.0]
Warmth: [0.0-1.0]
Fear: [0.0-1.0]
Unease: [0.0-1.0]
Shock: [0.0-1.0]
Futuristic: [0.0-1.0]
Technology: [0.0-1.0]
Wonder: [0.0-1.0]
Scale: [0.0-1.0]
Humor: [0.0-1.0]
Lightheartedness: [0.0-1.0]
Timing: [0.0-1.0]
Beauty: [0.0-1.0]"""
            
            # Prepare batch messages - one message per shot
            batch_messages = []
            for key_images in shot_key_images:
                if not key_images:
                    continue
                    
                # Create message with all key frames for this shot
                content = [{"type": "image", "image": img} for img in key_images]
                content.append({"type": "text", "text": prompt})
                
                batch_messages.append({
                    "role": "user",
                    "content": content
                })
            
            if not batch_messages:
                logger.error("No valid messages in batch")
                return [self._empty_analysis() for _ in shots_data]
            
            # Process all shots with batch processor
            logger.info(f"Processing {len(batch_messages)} shots in single forward pass")
            
            # Apply chat template to each message
            batch_texts = []
            for msg in batch_messages:
                text = self.processor.apply_chat_template(
                    [msg], tokenize=False, add_generation_prompt=True
                )
                batch_texts.append(text)
            
            # Extract all images for batch processing
            batch_images = []
            for msg in batch_messages:
                for content_item in msg["content"]:
                    if content_item.get("type") == "image":
                        batch_images.append(content_item["image"])
            
            # Process with model in batch
            inputs = self.processor(
                text=batch_texts,
                images=batch_images if batch_images else None,
                videos=None,
                padding=True,
                return_tensors="pt"
            )
            
            # Move to device (handle DataParallel wrapper)
            if isinstance(self.model, torch.nn.DataParallel):
                device = self.model.module.device
            else:
                device = self.model.device
            inputs = inputs.to(device)
            
            # Generate with mixed precision
            # Handle DataParallel wrapper - need to access .module.generate()
            model_to_use = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            
            logger.info(f"Running batched inference on {len(batch_texts)} shots")
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda'):
                        outputs = model_to_use.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=False
                        )
                else:
                    outputs = model_to_use.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )
            
            # Decode all responses
            output_ids = outputs[:, inputs.input_ids.shape[1]:]
            generated_texts = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            logger.info(f"Batch inference complete, parsing {len(generated_texts)} results")
            
            # Parse each result
            results = []
            for i, shot in enumerate(shots_data):
                if i < len(generated_texts):
                    parsed = self._parse_model_output(generated_texts[i])
                    
                    # Add audio fusion if available
                    if self.enable_audio_fusion and shot.get('audio_features'):
                        audio_context = self._process_audio_features(shot['audio_features'])
                        attributes = self._fuse_multimodal(parsed['attributes'], audio_context)
                    else:
                        attributes = parsed['attributes']
                    
                    results.append({
                        'caption': parsed['caption'],
                        'attributes': attributes,
                        'processing_mode': 'keyframes_parallel'
                    })
                else:
                    logger.warning(f"Missing result for shot {i}, using empty analysis")
                    results.append(self._empty_analysis())
            
            logger.info(f"Parallel batch processing complete for {len(results)} shots")
            return results
            
        except Exception as e:
            import traceback
            logger.error(f"Error in parallel batch processing: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.warning("Falling back to sequential processing")
            return [self.analyze_shot(shot) for shot in shots_data]
    
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
