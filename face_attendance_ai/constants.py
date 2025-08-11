# Configuration constants for Face Attendance System

# Directory paths
REGISTERED_FACES_DIR = "images/registered"
CAPTURED_FACES_DIR = "images/captured"
LOGS_DIR = "logs"

# File paths
CAPTURE_PATH = "images/captured/capture.jpg"
LOG_PATH = "logs/verified_faces.csv"

# Face verification settings
FACE_DISTANCE_THRESHOLD = 0.4
FACE_MODEL = "Facenet"
DETECTOR_BACKEND = "opencv"

# Image quality settings
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 220

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp')

# API settings
API_TITLE = "Face Attendance System"
API_VERSION = "1.0.0"
API_DESCRIPTION = "AI-powered face recognition attendance system"
