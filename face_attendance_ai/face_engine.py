import os
from deepface import DeepFace
from .constants import (
    REGISTERED_FACES_DIR, FACE_DISTANCE_THRESHOLD, 
    FACE_MODEL, DETECTOR_BACKEND, SUPPORTED_IMAGE_FORMATS
)

def verify_face(input_path, known_faces_dir=REGISTERED_FACES_DIR, model=FACE_MODEL):
    for person in os.listdir(known_faces_dir):
        # Skip non-image files like .gitkeep
        if not person.lower().endswith(SUPPORTED_IMAGE_FORMATS):
            continue
            
        ref_path = os.path.join(known_faces_dir, person)
        try:
            result = DeepFace.verify(img1_path=input_path,
                                     img2_path=ref_path,
                                     model_name=model,
                                     detector_backend=DETECTOR_BACKEND,
                                     enforce_detection=False)
            if result["verified"] and result["distance"] < FACE_DISTANCE_THRESHOLD:
                return {
                    "verified": True,
                    "matched_with": person,
                    "score": result["distance"]
                }
        except Exception as e:
            print(f"Error verifying {person}: {e}")
    return {"verified": False, "matched_with": None, "score": None}