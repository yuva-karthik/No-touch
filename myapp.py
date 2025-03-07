import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import os
import pyautogui
import time


# Configure PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# Initialize MediaPipe Holistic for better gesture recognition
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Initialize hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Global variables for gesture tracking
last_gesture = None
last_gesture_time = time.time()
gesture_hold_time = 0.5  # Time required to hold a gesture
action_cooldown = 0.5    # Time between actions
last_action_time = time.time()

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
    # Threshold angles for finger extension
    thresholds = [150, 160, 160, 160, 160]
    return [1 if angle > threshold else 0 for angle, threshold in zip(angles, thresholds)]

def get_gesture_name(fingers, angles):
    """Enhanced gesture recognition using both finger positions and angles"""
    gesture_dict = {
        (0, 0, 0, 0, 0): "✊ Fist",
        (1, 1, 1, 1, 1): "🖐 Open Palm",
        (0, 1, 0, 0, 0): "☝️ Pointing",
        (1, 1, 0, 0, 0): "✌️ Peace Sign",
        (1, 0, 0, 0, 0): "👍 Thumbs Up",
        (0, 1, 1, 1, 0): "🤘 Rock On",
        (1, 1, 1, 0, 0): "🤟 I Love You",
        (0, 1, 1, 0, 0): "🤞 Crossed Fingers",
        (1, 0, 0, 0, 1): "🤙 Call Me",
        (0, 0, 1, 0, 0): "🖕 Middle Finger"
    }
    
    # Additional angle-based checks for more accurate recognition
    gesture = gesture_dict.get(tuple(fingers), "Analyzing...")
    
    # Fine-tune gesture detection using angles
    if gesture == "👍 Thumbs Up" and angles[0] < 120:
        return "Analyzing..."
    elif gesture == "☝️ Pointing" and angles[1] < 150:
        return "Analyzing..."
    
    return gesture

def perform_action(current_gesture, hand_landmarks):
    """Perform actions based on gestures and transitions"""
    global last_gesture, last_gesture_time, last_action_time
    
    current_time = time.time()
    
    # Get index finger tip position for mouse control
    index_tip = hand_landmarks.landmark[8]
    screen_width, screen_height = pyautogui.size()
    mouse_x = int(index_tip.x * screen_width)
    mouse_y = int(index_tip.y * screen_height)
    
    # Check if enough time has passed since last action
    if current_time - last_action_time < action_cooldown:
        return
    
    # Handle gesture transitions and actions
    if current_gesture != last_gesture:
        last_gesture_time = current_time
        last_gesture = current_gesture
    elif current_time - last_gesture_time >= gesture_hold_time:
        # Perform actions based on gestures
        if current_gesture == "☝️ Pointing":
            # Move mouse with pointing finger
            pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)
            
        elif current_gesture == "✌️ Peace Sign":
            # Left click
            pyautogui.click()
            last_action_time = current_time
            
        elif current_gesture == "🤘 Rock On":
            # Right click
            pyautogui.rightClick()
            last_action_time = current_time
            
        elif current_gesture == "✊ Fist":
            # Double click
            pyautogui.doubleClick()
            last_action_time = current_time
            
        elif current_gesture == "🖐 Open Palm":
            # Standby/reset
            pass
            
        elif current_gesture == "🤙 Call Me":
            # Scroll up
            pyautogui.scroll(5)
            last_action_time = current_time
            
        elif current_gesture == "👍 Thumbs Up":
            # Scroll down
            pyautogui.scroll(-5)
            last_action_time = current_time

def detect_gesture(hand_landmarks):
    """Enhanced gesture detection with angle calculations"""
    # Convert landmarks to list for easier processing
    landmarks_list = [landmark for landmark in hand_landmarks.landmark]
    
    # Get extended fingers using angle-based detection
    fingers = get_extended_fingers(landmarks_list)
    
    # Calculate angles for more precise detection
    angles = calculate_finger_angles(landmarks_list)
    
    # Get gesture name using both fingers and angles
    gesture_name = get_gesture_name(fingers, angles)
    
    # Calculate hand direction
    wrist = landmarks_list[0]
    middle_tip = landmarks_list[12]
    direction = "Up" if middle_tip.y < wrist.y else "Down"
    
    # Perform action based on detected gesture
    perform_action(gesture_name, hand_landmarks)
    
    return fingers, gesture_name, direction

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Set up display window
cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1280, 720)

# Display instructions
def draw_instructions(frame):
    instructions = [
        "Gesture Controls:",
        "☝️ Pointing: Move Mouse",
        "✌️ Peace Sign: Left Click",
        "🤘 Rock On: Right Click",
        "✊ Fist: Double Click",
        "🖐 Open Palm: Standby",
        "🤙 Call Me: Scroll Up",
        "👍 Thumbs Up: Scroll Down",
        "Press 'q' to quit"
    ]
    
    y_offset = 200
    for instruction in instructions:
        cv2.putText(frame, instruction, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30

while True:
    success, frame = cap.read()
    if not success:
        break
        
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with both holistic and hands models
    holistic_results = holistic.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
    # Create a semi-transparent overlay for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw instructions
    draw_instructions(frame)
    
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Draw hand landmarks with improved styling
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Get enhanced gesture detection
            fingers, gesture_name, direction = detect_gesture(hand_landmarks)
            
            # Display current gesture and status
            cv2.putText(frame, f"Current Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Hand Direction: {direction}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display action status
            if time.time() - last_action_time < action_cooldown:
                cv2.putText(frame, "Action Cooldown...", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("Gesture Control", frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
holistic.close()
hands.close()
