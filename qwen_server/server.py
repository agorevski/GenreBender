"""
FastAPI server for Qwen2-VL multimodal shot analysis.
Provides endpoints for single and batch shot analysis with API key authentication.
"""

import os
import yaml
import logging
from typing import List, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

from model_loader import ModelLoader
from analyzer import MultimodalAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and analyzer
model = None
processor = None
analyzer = None
config = None
security = HTTPBearer()

# Load configuration
def load_config():
    """Load server configuration from YAML file.

    Reads the config.yaml file from the same directory as this module
    and parses it as YAML.

    Returns:
        dict: Configuration dictionary containing server, model, and
            processing settings.
    """
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager for FastAPI application lifespan events.

    Handles startup and shutdown of the server, including model loading
    and analyzer initialization.

    Args:
        app: The FastAPI application instance.

    Yields:
        None: Control is yielded to the application after startup completes.

    Raises:
        Exception: If model loading or analyzer initialization fails.
    """
    # Startup
    global model, processor, analyzer
    logger.info("Starting QWEN server...")
    
    try:
        # Load model
        model_loader = ModelLoader(config)
        model, processor = model_loader.load_model()
        
        # Initialize analyzer
        analyzer = MultimodalAnalyzer(model, processor, config)
        
        logger.info("Server ready!")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down server...")

# Create FastAPI app
app = FastAPI(
    title="Qwen2-VL Multimodal Shot Analyzer",
    description="Analyze video shots with multi-frame and audio features",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response
class ShotAnalysisRequest(BaseModel):
    shot_id: int = Field(..., description="Shot identifier")
    images: Optional[List[str]] = Field(None, description="List of base64 encoded images (keyframe mode)")
    video: Optional[str] = Field(None, description="Base64 encoded video file (video mode)")
    audio_features: Optional[Dict] = Field(None, description="Audio features dictionary")
    start_time: float = Field(..., description="Shot start time in seconds")
    end_time: float = Field(..., description="Shot end time in seconds")
    duration: float = Field(..., description="Shot duration in seconds")

class BatchAnalysisRequest(BaseModel):
    shots: List[ShotAnalysisRequest] = Field(..., description="List of shots to analyze")

class AttributeScores(BaseModel):
    suspense: float = Field(..., ge=0.0, le=1.0)
    darkness: float = Field(..., ge=0.0, le=1.0)
    ambiguity: float = Field(..., ge=0.0, le=1.0)
    emotional_tension: float = Field(..., ge=0.0, le=1.0)
    intensity: float = Field(..., ge=0.0, le=1.0)
    motion: float = Field(..., ge=0.0, le=1.0)

class AnalysisResponse(BaseModel):
    caption: str
    attributes: AttributeScores

class BatchAnalysisResponse(BaseModel):
    results: List[Dict]

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    gpu_count: Optional[int] = None
    gpu_memory_total: Optional[str] = None

# Authentication dependency
def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key from Authorization header.

    Args:
        credentials: HTTP authorization credentials containing the bearer token.

    Returns:
        HTTPAuthorizationCredentials: The credentials if validation succeeds.

    Raises:
        HTTPException: If the API key is invalid (401 Unauthorized).
    """
    expected_key = config['server']['api_key']
    
    if credentials.credentials != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials

# Health check endpoint (no auth required)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and return system information.

    Provides health status along with model and GPU information.
    No authentication required.

    Returns:
        HealthResponse: Dictionary containing server status, model name,
            device type, and optional GPU information (count and total memory).
    """
    import torch
    
    health_info = {
        "status": "healthy",
        "model": config['model']['name'],
        "device": config['model']['device']
    }
    
    if torch.cuda.is_available():
        health_info["gpu_count"] = torch.cuda.device_count()
        total_memory = sum(
            torch.cuda.get_device_properties(i).total_memory 
            for i in range(torch.cuda.device_count())
        ) / 1e9
        health_info["gpu_memory_total"] = f"{total_memory:.1f} GB"
    
    return health_info

# Single shot analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_shot(
    request: ShotAnalysisRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    """Analyze a single shot with multiple frames and audio features.

    Processes either keyframe images or video data along with optional
    audio features to generate a caption and attribute scores.
    Requires API key authentication.

    Args:
        request: Shot analysis request containing images or video data,
            audio features, and timing information.
        credentials: API key credentials from the Authorization header.

    Returns:
        AnalysisResponse: Analysis result containing a caption and
            attribute scores (suspense, darkness, ambiguity, etc.).

    Raises:
        HTTPException: 400 if request validation fails, 401 if unauthorized,
            500 if analysis fails.
    """
    try:
        # Validate request - must have either images or video
        if not request.images and not request.video:
            raise HTTPException(
                status_code=400, 
                detail="Must provide either 'images' (for keyframe mode) or 'video' (for video mode)"
            )
        
        if request.images and request.video:
            raise HTTPException(
                status_code=400,
                detail="Cannot provide both 'images' and 'video' - choose one mode"
            )
        
        if request.images and len(request.images) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Too many images (max 10 per shot)"
            )
        
        # Analyze shot
        shot_data = request.dict()
        result = analyzer.analyze_shot(shot_data)
        
        processing_mode = result.get('processing_mode', 'keyframes')
        logger.info(f"Analyzed shot {request.shot_id} using {processing_mode} mode")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing shot {request.shot_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch analysis endpoint
@app.post("/analyze_batch", response_model=BatchAnalysisResponse)
async def analyze_batch(
    request: BatchAnalysisRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    """Analyze multiple shots in a batch.

    Processes multiple shots either sequentially or in parallel depending
    on configuration. Validates all shots before processing.
    Requires API key authentication.

    Args:
        request: Batch analysis request containing a list of shots to analyze.
        credentials: API key credentials from the Authorization header.

    Returns:
        BatchAnalysisResponse: Results containing a list of analysis results,
            each with shot_id, caption, and attributes.

    Raises:
        HTTPException: 400 if batch validation fails or exceeds max size,
            401 if unauthorized, 500 if analysis fails.
    """
    try:
        # Validate batch size
        max_batch_size = config['processing']['max_batch_size']
        if len(request.shots) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum ({max_batch_size})"
            )
        
        # Validate each shot before processing
        validation_errors = []
        logger.info(f"Validating batch of {len(request.shots)} shots")
        
        for idx, shot_request in enumerate(request.shots):
            shot_id = shot_request.shot_id
            has_images = shot_request.images is not None
            has_video = shot_request.video is not None
            images_count = len(shot_request.images) if shot_request.images else 0
            
            logger.debug(f"Shot {shot_id} (idx {idx}): has_images={has_images}, has_video={has_video}, images_count={images_count}")
            
            # Check that shot has either images or video
            if not shot_request.images and not shot_request.video:
                error = f"Shot {shot_id} (index {idx}): Must provide either 'images' or 'video'"
                validation_errors.append(error)
                logger.warning(error)
            elif shot_request.images and shot_request.video:
                error = f"Shot {shot_id} (index {idx}): Cannot provide both 'images' and 'video'"
                validation_errors.append(error)
                logger.warning(error)
            elif shot_request.images and len(shot_request.images) == 0:
                error = f"Shot {shot_id} (index {idx}): 'images' array is empty"
                validation_errors.append(error)
                logger.warning(error)
            elif shot_request.images and len(shot_request.images) > 10:
                error = f"Shot {shot_id} (index {idx}): Too many images ({len(shot_request.images)}, max 10 per shot)"
                validation_errors.append(error)
                logger.warning(error)
        
        if validation_errors:
            error_msg = f"Batch validation failed with {len(validation_errors)} errors:\n" + "\n".join(validation_errors[:5])
            if len(validation_errors) > 5:
                error_msg += f"\n... and {len(validation_errors) - 5} more errors"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Batch validation passed for {len(request.shots)} shots")
        
        # Check if parallel batching is enabled
        use_parallel = config['processing'].get('use_parallel_batching', False)
        
        if use_parallel:
            # Use parallel batch processing (Phase 2)
            logger.info(f"Using parallel batch processing for {len(request.shots)} shots")
            shots_data = [shot_request.dict() for shot_request in request.shots]
            batch_results = analyzer.analyze_batch_parallel(shots_data)
            
            # Format results
            results = []
            for i, shot_request in enumerate(request.shots):
                if i < len(batch_results):
                    results.append({
                        'shot_id': shot_request.shot_id,
                        'caption': batch_results[i]['caption'],
                        'attributes': batch_results[i]['attributes']
                    })
                else:
                    logger.warning(f"Missing result for shot {shot_request.shot_id}")
                    results.append({
                        'shot_id': shot_request.shot_id,
                        'caption': 'Analysis failed',
                        'attributes': {}
                    })
        else:
            # Use sequential processing (legacy)
            logger.info(f"Using sequential processing for {len(request.shots)} shots")
            results = []
            for shot_request in request.shots:
                shot_data = shot_request.dict()
                result = analyzer.analyze_shot(shot_data)
                
                results.append({
                    'shot_id': shot_request.shot_id,
                    'caption': result['caption'],
                    'attributes': result['attributes']
                })
        
        logger.info(f"Analyzed batch of {len(results)} shots")
        
        return {"results": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Return API information and available endpoints.

    Provides basic information about the API including its name, version,
    available endpoints, and authentication requirements.

    Returns:
        dict: API information containing name, version, endpoints mapping,
            and authentication details.
    """
    return {
        "name": "Qwen2-VL Multimodal Shot Analyzer",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "analyze_batch": "POST /analyze_batch"
        },
        "authentication": "Bearer token required for analysis endpoints"
    }

# Run server
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Qwen2-VL Server')
    parser.add_argument('--port', type=int, default=None,
                       help='Port to run server on (overrides config)')
    parser.add_argument('--host', type=str, default=None,
                       help='Host to bind to (overrides config)')
    args = parser.parse_args()
    
    # Get configuration with command line overrides
    host = args.host if args.host else config['server']['host']
    port = args.port if args.port else config['server']['port']
    reload = config['server'].get('reload', False)
    workers = config['server'].get('workers', 1)
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers
    )
