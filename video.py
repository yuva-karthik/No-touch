# video.py
import cv2

def init_video():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Gesture Control", 1280, 720)
    return cap

def release_video(cap):
    cap.release()
    cv2.destroyAllWindows()
