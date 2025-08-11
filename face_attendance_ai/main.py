from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from .face_engine import verify_face
from .utils import capture_face, log_verification, basic_spoof_check
from .constants import REGISTERED_FACES_DIR, API_TITLE, API_VERSION, API_DESCRIPTION
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class FaceVerifyResponse(BaseModel):
    verified: bool
    matched_with: str | None
    score: float | None

class RegisterRequest(BaseModel):
    name: str

class RegisterResponse(BaseModel):
    success: bool
    message: str
    name: str | None

class ClientVerifyResponse(BaseModel):
    match: bool
    confidence: float

@app.get("/")
def health_check():
    return {"status": "AI module is live"}

@app.post("/api/verify", response_model=FaceVerifyResponse)
@limiter.limit("5/minute")
def verify(request: Request):
    try:
        if not os.path.exists(REGISTERED_FACES_DIR) or not os.listdir(REGISTERED_FACES_DIR):
            logger.error("No registered faces found")
            raise HTTPException(status_code=400, detail="No registered faces found. Please add faces to images/registered/ directory.")
        
        logger.info("Starting face capture...")
        path = capture_face()
        
        logger.info("Performing spoof check...")
        if not basic_spoof_check(path):
            logger.warning("Spoof detection failed")
            return {"verified": False, "matched_with": None, "score": 0.0}

        logger.info("Starting face verification...")
        result = verify_face(path)
        
        log_verification(result['matched_with'], result['verified'], result['score'])
        logger.info(f"Verification result: {result}")
        
        return result

    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Required files or directories not found")
    except Exception as e:
        logger.error(f"Verification error: {str(e)}")
        return {"verified": False, "matched_with": None, "score": 0.0, "error": "Internal server error"}

@app.post("/api/register", response_model=RegisterResponse)
@limiter.limit("3/minute")
def register_face(request: RegisterRequest, req: Request):
    """Register a new face by capturing from webcam"""
    try:
        if not request.name or len(request.name.strip()) == 0:
            raise HTTPException(status_code=400, detail="Name cannot be empty")
        
        clean_name = request.name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_")
        
        registered_path = f"images/registered/{clean_name}.jpg"
        if os.path.exists(registered_path):
            logger.warning(f"Face already registered for: {clean_name}")
            raise HTTPException(status_code=400, detail=f"Face already registered for {request.name}")
        
        logger.info(f"Starting face registration for: {request.name}")
        captured_path = capture_face()
        
        logger.info("Performing spoof check for registration...")
        if not basic_spoof_check(captured_path):
            logger.warning("Spoof detection failed during registration")
            raise HTTPException(status_code=400, detail="Face capture failed spoof detection. Please try again with better lighting.")
        
        import shutil
        shutil.copy2(captured_path, registered_path)
        logger.info(f"Face registered successfully: {registered_path}")
        
        return {
            "success": True,
            "message": f"Face registered successfully for {request.name}",
            "name": request.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        raise HTTPException(status_code=500, detail="Registration failed due to internal error")

@app.get("/api/registered")
def list_registered_faces():
    """List all registered faces"""
    try:
        if not os.path.exists("images/registered"):
            return {"registered_faces": []}
        
        faces = []
        for filename in os.listdir("images/registered"):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                name = os.path.splitext(filename)[0].replace("_", " ")
                faces.append({
                    "name": name,
                    "filename": filename
                })
        
        return {"registered_faces": faces, "count": len(faces)}
        
    except Exception as e:
        logger.error(f"Error listing registered faces: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list registered faces")

@app.post("/verify-face", response_model=ClientVerifyResponse)
@limiter.limit("10/minute")
async def verify_face_from_photo(request: Request, file: UploadFile = File(...)):
    """Client-specific endpoint: Accept photo from frontend and return match/confidence"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check if registered faces exist
        if not os.path.exists(REGISTERED_FACES_DIR) or not os.listdir(REGISTERED_FACES_DIR):
            logger.error("No registered faces found")
            raise HTTPException(status_code=400, detail="No registered employees found in system")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        try:
            # Basic spoof detection
            logger.info("Performing spoof check on uploaded image...")
            if not basic_spoof_check(temp_path):
                logger.warning("Spoof detection failed")
                return {"match": False, "confidence": 0.0}

            # Face verification
            logger.info("Starting face verification...")
            result = verify_face(temp_path)
            
            # Convert to client format
            if result["verified"]:
                # Convert distance to confidence percentage (lower distance = higher confidence)
                confidence = max(0.0, min(100.0, (1 - result["score"]) * 100))
                logger.info(f"Match found: {result['matched_with']} with confidence {confidence:.1f}%")
                
                # Log the result
                log_verification(result['matched_with'], True, result['score'])
                
                return {
                    "match": True,
                    "confidence": round(confidence, 1)
                }
            else:
                logger.info("No match found")
                log_verification(None, False, None)
                return {
                    "match": False,
                    "confidence": 0.0
                }
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face verification error: {str(e)}")
        raise HTTPException(status_code=500, detail="Face verification failed")