"""
FastAPI Backend Application
---------------------------
Provides API endpoints for Side-Scan Sonar Object Detection and serves the frontend prototype.
"""

import os
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.model_service import detect_sonar_objects, REAL_MODEL_ENABLED

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLE_DIR = BASE_DIR / "sample_images"

# Ensure sample images exist
if not SAMPLE_DIR.exists() or len(list(SAMPLE_DIR.glob("*.jpg"))) == 0:
    from backend.sample_generator import generate_sample_images
    generate_sample_images(str(SAMPLE_DIR))

app = FastAPI(
    title="Underwater Sonar Object Detection API",
    description="AI-powered Side-Scan Sonar object detection prototype",
    version="1.0.0"
)

# Enable CORS for local testing or cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum allowed file size: 25 MB
MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".jfif", ".gif"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/pjpeg", "image/bmp", "image/tiff", "image/gif", "application/octet-stream"}


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint providing model mode and status."""
    return {
        "status": "online",
        "service": "Underwater Sonar Object Detection",
        "model_mode": "REAL_MODEL" if REAL_MODEL_ENABLED else "MOCK_DEMO_READY",
        "supported_formats": list(ALLOWED_EXTENSIONS)
    }


@app.get("/api/samples")
async def get_sample_images():
    """Returns metadata for pre-bundled sonar demo test images."""
    samples = [
        {
            "id": "small_object",
            "name": "Small Object Detection",
            "description": "Compact high-density acoustic contact with tight shadow signature",
            "filename": "sample_small_object.jpg",
            "url": "/sample_images/sample_small_object.jpg"
        },
        {
            "id": "unknown_anomaly",
            "name": "Unknown Anomaly Detection",
            "description": "Unidentified seabed acoustic reverberation and perturbation contour",
            "filename": "sample_unknown_anomaly.jpg",
            "url": "/sample_images/sample_unknown_anomaly.jpg"
        },
        {
            "id": "ghost_net",
            "name": "Ghost Net Detection",
            "description": "Discarded commercial netting filament drape across seafloor",
            "filename": "sample_ghost_net.jpg",
            "url": "/sample_images/sample_ghost_net.jpg"
        }
    ]
    return {"samples": samples}


@app.post("/predict")
async def predict_sonar_image(file: UploadFile = File(...)):
    """
    Main detection endpoint.
    Accepts an uploaded image file (JPG, JPEG, PNG, WEBP, BMP, TIFF).
    Returns detection results in standard JSON format:
    {
      "detections": [
        {
          "class": "small_object",
          "confidence": 0.94,
          "x": 420,
          "y": 180,
          "width": 100,
          "height": 80
        }
      ]
    }
    """
    # 1. Validate file extension (if present)
    filename = file.filename or "upload.jpg"
    ext = os.path.splitext(filename)[1].lower()
    
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload an image file (JPG, PNG, WEBP, BMP, or TIFF)."
        )
        
    # 2. Read content and validate file size
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read uploaded file: {str(e)}"
        )
        
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )
        
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / (1024*1024):.1f} MB). Maximum allowed size is 25 MB."
        )
        
    # 3. Perform object detection inference via model_service
    try:
        results = detect_sonar_objects(content, filename)
        return JSONResponse(content=results)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection inference failed: {str(e)}"
        )


# Mount sample images folder
if SAMPLE_DIR.exists():
    app.mount("/sample_images", StaticFiles(directory=str(SAMPLE_DIR)), name="sample_images")

# Mount frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend_static")

    @app.get("/")
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
