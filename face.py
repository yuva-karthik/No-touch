# face.py
import face_recognition
import cv2
import numpy as np

# Path to the authorized face image
AUTHORIZED_FACE_PATH = r"C:\Users\Sonu\Pictures\Camera Roll\mypic2.py.jpg"

# Load the authorized image and compute its face encoding
authorized_image = face_recognition.load_image_file(AUTHORIZED_FACE_PATH)
authorized_encodings = face_recognition.face_encodings(authorized_image)

if len(authorized_encodings) > 0:
    authorized_face_encoding = authorized_encodings[0]
else:
    authorized_face_encoding = None
    print("No face found in the authorized image!")

def is_authorized_face(frame, tolerance=0.6):
    """
    Checks if the current frame contains a face matching the authorized face.
    
    Parameters:
      frame: the current frame in BGR color space (as from OpenCV)
      tolerance: how strict the matching should be (default is 0.6)
    
    Returns:
      True if an authorized face is found, otherwise False.
    """
    if authorized_face_encoding is None:
        return False

    # Convert the frame from BGR (OpenCV) to RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Find face locations and encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # Compare each face found in the frame with the authorized face encoding
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces([authorized_face_encoding], face_encoding, tolerance=tolerance)
        if True in matches:
            return True

    return False