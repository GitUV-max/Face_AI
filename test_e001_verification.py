#!/usr/bin/env python3
"""
Test script to verify E001 face recognition between registered and captured images
"""

import sys
import os
sys.path.append('.')

from face_attendance_ai.face_engine import verify_face
from face_attendance_ai.utils import basic_spoof_check
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_e001_verification():
    """Test face verification between registered E001 and captured E001"""
    
    registered_path = "images/registered/E001.jpg"
    captured_path = "images/captured/E001.jpg"
    
    print("=" * 60)
    print("🧪 TESTING E001 FACE VERIFICATION")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(registered_path):
        print(f"❌ Registered image not found: {registered_path}")
        return False
        
    if not os.path.exists(captured_path):
        print(f"❌ Captured image not found: {captured_path}")
        return False
    
    print(f"✅ Registered image: {registered_path} ({os.path.getsize(registered_path)} bytes)")
    print(f"✅ Captured image: {captured_path} ({os.path.getsize(captured_path)} bytes)")
    print()
    
    # Test 1: Spoof check on captured image
    print("🔍 Test 1: Spoof Detection on Captured Image")
    print("-" * 40)
    try:
        spoof_result = basic_spoof_check(captured_path)
        print(f"Spoof check result: {'✅ PASSED' if spoof_result else '❌ FAILED'}")
    except Exception as e:
        print(f"❌ Spoof check error: {e}")
        spoof_result = False
    print()
    
    # Test 2: Face verification
    print("🔍 Test 2: Face Verification (Captured vs Registered)")
    print("-" * 40)
    try:
        verification_result = verify_face(captured_path, "images/registered")
        
        print(f"Verification Result:")
        print(f"  - Verified: {verification_result['verified']}")
        print(f"  - Matched with: {verification_result['matched_with']}")
        print(f"  - Distance Score: {verification_result['score']}")
        
        if verification_result['verified']:
            # Convert to client format
            confidence = max(0.0, min(100.0, (1 - verification_result['score']) * 100))
            print(f"  - Confidence: {confidence:.1f}%")
            print()
            print("RESPONSE:")
            print(f"   {{\"match\": true, \"confidence\": {confidence:.1f}}}")
        else:
            print()
            print("RESPONSE:")
            print(f"   {{\"match\": false, \"confidence\": 0.0}}")
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        verification_result = {"verified": False, "matched_with": None, "score": None}
    
    print()
    print("=" * 60)
    
    # Summary
    success = spoof_result and verification_result['verified']
    if success:
        print("🎉 OVERALL TEST RESULT: ✅ SUCCESS")
        print("   - E001 images match successfully")
        print("   - System is working correctly")
    else:
        print("❌ OVERALL TEST RESULT: ❌ FAILED")
        if not spoof_result:
            print("   - Spoof detection failed")
        if not verification_result['verified']:
            print("   - Face verification failed")
    
    print("=" * 60)
    return success

if __name__ == "__main__":
    test_e001_verification()
