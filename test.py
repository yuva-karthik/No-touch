from deepface import DeepFace
import cv2

# Load the reference image for authentication
reference_image = cv2.imread('mypic.jpg')

# Function to authenticate the face
def authenticate_face(frame):
    try:
        # Use DeepFace to verify the face in the current frame against the reference image
        result = DeepFace.verify(frame, reference_image, model_name='VGG-Face')
        return result['verified']
    except Exception as e:
        print(f"Error during face recognition: {e}")
        return False

# In your main loop or function where gestures are processed
current_frame = ...  # Capture the current frame from the video feed
if authenticate_face(current_frame):
    print("Authentication successful. Access granted.")
else:
    print("Authentication failed. Access denied.")