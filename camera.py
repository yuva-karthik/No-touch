import cv2

def find_all_available_camera_indices():
    available_cameras = []
    # Check camera indices from 0 to 9 (you can increase this range if needed)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()  # Release the camera after checking
    return available_cameras

# Example usage
camera_indices = find_all_available_camera_indices()
print("Available camera indices:", camera_indices)
