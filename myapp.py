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
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

# Initialize hands model
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Global variables for gesture tracking
last_gesture = None
last_gesture_time = time.time()
gesture_hold_time = 0.3
action_cooldown = 0.2
last_action_time = time.time()

# Mouse control settings
MOUSE_SENSITIVITY = 2.0  # Adjust this value to change mouse movement sensitivity
last_hand_position = None
smooth_factor = 0.5  # Smoothing factor for mouse movement (0 to 1)

# Mini player settings
MINI_PLAYER_WIDTH = 320  # Width of mini player
MINI_PLAYER_HEIGHT = 180  # Height of mini player (16:9 ratio)
MINI_PLAYER_MARGIN = 20  # Margin from the edges

# Create separate window for mini player
cv2.namedWindow("Mini Player", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mini Player", MINI_PLAYER_WIDTH, MINI_PLAYER_HEIGHT)

# Set mini player window position (bottom-right of screen)
screen_width, screen_height = pyautogui.size()
cv2.moveWindow("Mini Player", 
              screen_width - MINI_PLAYER_WIDTH - 50,
              screen_height - MINI_PLAYER_HEIGHT - 100)

# Video file path - replace with your video file
VIDEO_PATH = "path_to_your_video.mp4"  # You'll need to set this to your video file path

# Initialize video player
try:
    video_player = cv2.VideoCapture(VIDEO_PATH)
    video_loaded = True
except:
    video_loaded = False
    print("No video file found. Only webcam feed will be shown.")

def resize_frame_with_aspect_ratio(frame, width=None, height=None):
    """Resize frame maintaining aspect ratio"""
    (h, w) = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

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

def get_gesture_name(fingers, angles):
    """Enhanced gesture recognition using both finger positions and angles"""
    gesture_dict = {
        (0, 0, 0, 0, 0): "✊ Fist",
        (1, 1, 1, 1, 1): "🖐 Open Palm",
        (1, 1, 1, 0, 0): "🤟 I Love You",
        (0, 1, 1, 1, 0): "🤘 Yo",
        (1, 0, 0, 0, 0): "👍 Thumbs Up",
        (0, 0, 0, 0, 1): "👎 Thumbs Down"
    }
    
    # Additional angle-based checks for more accurate recognition
    gesture = gesture_dict.get(tuple(fingers), "Analyzing...")
    
    # Reduced angle thresholds for easier gesture detection
    if gesture == "👍 Thumbs Up" and angles[0] < 100:
        return "Analyzing..."
    elif gesture == "👎 Thumbs Down" and angles[0] < 100:
        return "Analyzing..."
    
    return gesture

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
    
    # Calculate thumb direction for scrolling
    thumb_tip = landmarks_list[4]  # Thumb tip
    thumb_base = landmarks_list[2]  # Thumb base
    thumb_direction = "Up" if thumb_tip.y < thumb_base.y else "Down"
    
    # Perform action based on detected gesture and thumb direction
    perform_action(gesture_name, hand_landmarks, thumb_direction)
    
    return fingers, gesture_name, thumb_direction

def perform_action(current_gesture, hand_landmarks, thumb_direction):
    """Perform actions based on gestures and transitions"""
    global last_gesture, last_gesture_time, last_action_time, last_hand_position
    
    current_time = time.time()
    
    # Get middle finger tip position for mouse control
    middle_tip = hand_landmarks.landmark[12]
    
    # Handle gesture transitions and actions
    if current_gesture != last_gesture:
        last_gesture_time = current_time
        last_gesture = current_gesture
        # Reset hand position when gesture changes
        last_hand_position = None
    elif current_time - last_gesture_time >= gesture_hold_time:
        # Perform actions based on gestures
        if current_gesture == "🖐 Open Palm":
            # Move mouse based on relative hand movement
            if last_hand_position is not None:
                # Calculate movement delta
                delta_x = (middle_tip.x - last_hand_position[0]) * screen_width * MOUSE_SENSITIVITY
                delta_y = (middle_tip.y - last_hand_position[1]) * screen_height * MOUSE_SENSITIVITY
                
                # Get current mouse position
                current_x, current_y = pyautogui.position()
                
                # Apply smoothing to the movement
                new_x = current_x + int(delta_x * smooth_factor)
                new_y = current_y + int(delta_y * smooth_factor)
                
                # Ensure coordinates stay within screen bounds
                new_x = max(0, min(new_x, screen_width - 1))
                new_y = max(0, min(new_y, screen_height - 1))
                
                # Move mouse
                pyautogui.moveTo(new_x, new_y)
            
            # Update last hand position
            last_hand_position = (middle_tip.x, middle_tip.y)
            
        elif current_gesture == "🤟 I Love You":
            # Left click
            pyautogui.click()
            last_action_time = current_time
            
        elif current_gesture == "🤘 Yo":
            # Right click
            pyautogui.rightClick()
            last_action_time = current_time
            
        elif current_gesture == "✊ Fist":
            # Standby/reset
            last_hand_position = None
            pass
            
        elif current_gesture == "👍 Thumbs Up":
            # Scroll based on thumb direction
            scroll_amount = 20 if thumb_direction == "Up" else -20
            pyautogui.scroll(scroll_amount)
            last_action_time = current_time

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Set up display window
cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1280, 720)

# Display instructions
def draw_instructions(frame):
    instructions = [
        "Gesture Controls:",
        "🖐 Open Palm: Move Cursor",
        "✊ Fist: Standby",
        "🤟 I Love You: Left Click",
        "🤘 Yo: Right Click",
        "👍 Thumbs Up + Direction: Scroll Up/Down",
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
        
    # Flip the main webcam frame horizontally
    frame = cv2.flip(frame, 1)
        
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
    
    # Handle mini video player in separate window
    if video_loaded:
        ret, video_frame = video_player.read()
        if ret:
            # Flip the video frame horizontally
            video_frame = cv2.flip(video_frame, 1)
            
            # Resize video frame for mini player
            video_frame = resize_frame_with_aspect_ratio(video_frame, 
                                                       width=MINI_PLAYER_WIDTH)
            
            # Add border and title to mini player
            # Add black background padding
            padding = np.zeros((MINI_PLAYER_HEIGHT + 40, MINI_PLAYER_WIDTH + 4, 3), dtype=np.uint8)
            # Add title
            cv2.putText(padding, "Mini Player", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Place video frame in padding
            h, w = video_frame.shape[:2]
            y_offset = 35
            x_offset = 2
            padding[y_offset:y_offset+h, x_offset:x_offset+w] = video_frame
            
            # Display mini player in separate window
            cv2.imshow("Mini Player", padding)
            
            # Ensure window stays on top by periodically moving it
            cv2.moveWindow("Mini Player", 
                         screen_width - MINI_PLAYER_WIDTH - 50,
                         screen_height - MINI_PLAYER_HEIGHT - 100)
        else:
            # Reset video to beginning when it ends
            video_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
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
            fingers, gesture_name, thumb_direction = detect_gesture(hand_landmarks)
            
            # Display current gesture and status
            cv2.putText(frame, f"Current Gesture: {gesture_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Thumb Direction: {thumb_direction}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display action status
            if time.time() - last_action_time < action_cooldown:
                cv2.putText(frame, "Action Cooldown...", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the main frame
    cv2.imshow("Gesture Control", frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
if video_loaded:
    video_player.release()
cv2.destroyAllWindows()
holistic.close()
hands.close()
