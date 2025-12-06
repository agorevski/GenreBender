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
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
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
    images: List[str] = Field(..., description="List of base64 encoded images")
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
    """
    Verify API key from Authorization header.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Credentials if valid
        
    Raises:
        HTTPException: If API key is invalid
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
    """
    Health check endpoint to verify server is running.
    No authentication required.
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
    """
    Analyze a single shot with multiple frames and audio features.
    Requires API key authentication.
    
    Args:
        request: Shot analysis request with images and audio features
        credentials: API key credentials
        
    Returns:
        Analysis result with caption and attribute scores
    """
    try:
        # Validate request
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        if len(request.images) > 10:
            raise HTTPException(
                status_code=400, 
                detail="Too many images (max 10 per shot)"
            )
        
        # Analyze shot
        shot_data = request.dict()
        result = analyzer.analyze_shot(shot_data)
        
        logger.info(f"Analyzed shot {request.shot_id}")
        
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
    """
    Analyze multiple shots in a batch.
    Requires API key authentication.
    
    Args:
        request: Batch analysis request with multiple shots
        credentials: API key credentials
        
    Returns:
        Batch analysis results
    """
    try:
        # Validate batch size
        max_batch_size = config['processing']['max_batch_size']
        if len(request.shots) > max_batch_size:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size exceeds maximum ({max_batch_size})"
            )
        
        # Analyze each shot
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
    """
    Root endpoint with API information.
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
    host = config['server']['host']
    port = config['server']['port']
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
