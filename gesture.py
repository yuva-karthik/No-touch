# gesture.py
import numpy as np
from math import sqrt
from config import *

def calculate_finger_angles(landmarks):
    """Calculate angles between finger joints"""
    angles = []
    # Define finger joint connections
    finger_joints = [
        [2, 3, 4],    # Thumb
        [5, 6, 8],    # Index
        [9, 10, 12],  # Middle
        [13, 14, 16], # Ring
        [17, 18, 20]  # Pinky
    ]
    
    for joint in finger_joints:
        p1 = np.array([landmarks[joint[0]].x, landmarks[joint[0]].y])
        p2 = np.array([landmarks[joint[1]].x, landmarks[joint[1]].y])
        p3 = np.array([landmarks[joint[2]].x, landmarks[joint[2]].y])
        
        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2
        
        # Calculate angle
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(np.degrees(angle))
    
    return angles

def get_extended_fingers(landmarks):
    """Get more accurate finger positions using angles"""
    angles = calculate_finger_angles(landmarks)
    # Reduced thresholds for easier finger detection
    thresholds = [130, 140, 140, 140, 140]
    return [1 if angle > threshold else 0 for angle, threshold in zip(angles, thresholds)]

def get_gesture_name(fingers, angles, thumb_direction):
    """Enhanced gesture recognition including thumb direction"""
    gesture_dict = {
    (0, 0, 0, 0, 0): "✊ Fist",          # All fingers closed
    (1, 1, 1, 1, 1): "🖐 Open Palm",    # All fingers open
    (1, 1, 1, 0, 0): "🤟 I Love You",   # Thumb, index, and middle fingers open
    (0, 1, 1, 1, 0): "🤘 Yo",           # Index, middle, and ring fingers open
    (0, 0, 0, 0, 1): "👎 Thumbs Down",  # Only thumb open (downward)
    (1, 0, 0, 0, 0): "👍 Thumbs Up",    # Only thumb open (upward)
    (0, 1, 1, 0, 0): "🤙 Call Me"       # Index and middle fingers open (like a "call me" gesture)
}
    # Special case for Thumbs Up (up/down)
    if tuple(fingers) == (1, 0, 0, 0, 0):
        if thumb_direction == "Up":
            return "👍 Thumbs Up"
        else:
            return "👍⬇ Thumbs Up Down"  # For backspace
    
    return gesture_dict.get(tuple(fingers), "Analyzing...")

def detect_gesture(hand_landmarks):
    """Enhanced gesture detection with angle calculations"""
    # Convert landmarks to list for easier processing
    landmarks_list = [landmark for landmark in hand_landmarks.landmark]
    
    # Get extended fingers using angle-based detection
    fingers = get_extended_fingers(landmarks_list)
    
    # Calculate angles for more precise detection
    angles = calculate_finger_angles(landmarks_list)
    
    # Get thumb direction for distinguishing up/down thumbs up
    thumb_tip = landmarks_list[4]  # Thumb tip
    thumb_base = landmarks_list[2]  # Thumb base
    thumb_direction = "Up" if thumb_tip.y < thumb_base.y else "Down"
    
    # Get gesture name using both fingers and angles
    gesture_name = get_gesture_name(fingers, angles, thumb_direction)
    
    return fingers, gesture_name, thumb_direction
