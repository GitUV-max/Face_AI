import os
import cv2
from deepface import DeepFace

def test_face_verification():
    print("=== Face Verification Debug Test ===")
    
    # Check registered faces
    registered_dir = "images/registered"
    if not os.path.exists(registered_dir):
        print("❌ Registered faces directory doesn't exist")
        return
    
    registered_faces = [f for f in os.listdir(registered_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"✅ Found {len(registered_faces)} registered faces: {registered_faces}")
    
    # Test image capture
    print("\n=== Testing Image Capture ===")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return
    
    print("📷 Webcam accessible. Taking a test photo...")
    ret, frame = cap.read()
    if ret:
        test_path = "images/captured/debug_test.jpg"
        cv2.imwrite(test_path, frame)
        print(f"✅ Test image saved: {test_path}")
    else:
        print("❌ Failed to capture image")
        cap.release()
        return
    cap.release()
    
    # Test face detection on captured image
    print("\n=== Testing Face Detection ===")
    try:
        faces = DeepFace.extract_faces(image_path=test_path, detector_backend='opencv', enforce_detection=False)
        print(f"✅ Detected {len(faces)} faces in captured image")
    except Exception as e:
        print(f"❌ Face detection failed: {e}")
        return
    
    # Test face verification against registered faces
    print("\n=== Testing Face Verification ===")
    for registered_face in registered_faces:
        registered_path = os.path.join(registered_dir, registered_face)
        try:
            result = DeepFace.verify(
                img1_path=test_path,
                img2_path=registered_path,
                model_name="Facenet",
                detector_backend='opencv',
                enforce_detection=False
            )
            print(f"📊 {registered_face}: verified={result['verified']}, distance={result['distance']:.4f}")
            
            # Apply our threshold
            if result["verified"] and result["distance"] < 0.4:
                print(f"✅ MATCH FOUND: {registered_face} (distance: {result['distance']:.4f})")
            else:
                print(f"❌ No match: {registered_face} (distance: {result['distance']:.4f}, threshold: 0.4)")
                
        except Exception as e:
            print(f"❌ Verification failed for {registered_face}: {e}")

if __name__ == "__main__":
    test_face_verification()
