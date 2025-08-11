import cv2
import os
from datetime import datetime
import pandas as pd
from deepface import DeepFace
import logging
from .constants import (
    CAPTURE_PATH, LOG_PATH, MIN_BRIGHTNESS, MAX_BRIGHTNESS,
    CAPTURED_FACES_DIR, LOGS_DIR
)

logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs(os.path.dirname(CAPTURE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def capture_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam not accessible")
        raise Exception("Webcam not accessible")
    
    print("[INFO] Press SPACE to capture...")
    while True:
        ret, frame = cap.read()
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            cv2.imwrite(CAPTURE_PATH, frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return CAPTURE_PATH

def log_verification(name, verified, score):
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = pd.DataFrame([[name, verified, score, ts]],
                           columns=["Name", "Verified", "Score", "Timestamp"])
        if os.path.exists(LOG_PATH):
            log.to_csv(LOG_PATH, mode='a', header=False, index=False)
        else:
            log.to_csv(LOG_PATH, index=False)
    except Exception as e:
        logger.error(f"Failed to log verification: {str(e)}")

def basic_spoof_check(image_path):
    """Basic spoof detection - ensures face is detected and image quality is acceptable"""
    try:
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return False
            
        faces = DeepFace.extract_faces(image_path=image_path, detector_backend='opencv', enforce_detection=False)
        
        if not faces or len(faces) == 0:
            logger.warning("No faces detected in captured image")
            return False
            
        img = cv2.imread(image_path)
        if img is None:
            logger.error("Failed to read captured image")
            return False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        
        if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
            logger.warning(f"Image brightness suspicious: {brightness}")
            return False
            
        logger.info("Basic spoof check passed")
        return True
        
    except Exception as e:
        logger.error(f"Spoof check failed: {str(e)}")
        return False