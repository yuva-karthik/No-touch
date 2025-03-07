import cv2
import mediapipe as mp
import numpy as np
from math import sqrt
import os
import pyautogui
import time
import speech_recognition as sr
import win32gui
import win32api
import win32con

# Configure PyAutoGUI
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Initialize speech recognition
recognizer = sr.Recognizer()
try:
    microphone = sr.Microphone()
    print("Speech recognition initialized successfully")
except Exception as e:
    print(f"Error initializing microphone: {e}")
    print("Please ensure you have a working microphone and PyAudio is installed")
    print("Install PyAudio with: pip install pyaudio")

# Text input mode settings
is_text_input_mode = False
last_text_input_check = time.time()
text_input_check_interval = 0.5  # Check for text box every 0.5 seconds

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
primary_hand_id = None  # Track the ID of the primary hand
primary_hand_time = 0  # Time when primary hand was last active
hand_tracking_history = {}  # Track when each hand first appeared
hand_tracking_timeout = 2.0  # Time before removing hand from history

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

def is_cursor_in_text_field():
    """Enhanced check if cursor is in a text input field"""
    try:
        # Get the foreground window and focused control
        foreground_window = win32gui.GetForegroundWindow()
        focused_control = win32gui.GetFocus()
        
        # Get class names
        window_class = win32gui.GetClassName(foreground_window)
        control_class = win32gui.GetClassName(focused_control) if focused_control else ""
        
        # Expanded list of text input class names
        text_input_classes = [
            'Edit',  # Standard Windows text box
            'RichEdit',  # Rich text box
            'RichEdit20W',  # Modern rich text box
            'RichEdit20A',  # Another rich text variant
            'TextBox',  # Generic text box
            'RICHEDIT50W',  # Windows 10 rich edit
            'Chrome_RenderWidgetHostHWND',  # Chrome browser
            'MozillaWindowClass',  # Firefox
            'Scintilla',  # Code editors
            'Notepad',  # Notepad
            'WordPadClass',  # WordPad
            'EXCEL7',  # Excel cells
            'OpusApp',  # Word documents
            'SUMATRA_PDF_FRAME'  # PDF editor fields
        ]

        # Common window classes that might contain text fields
        container_classes = [
            'Chrome_WidgetWin_1',  # Chrome windows
            'MozillaWindowClass',  # Firefox windows
            'Notepad',
            'WordPadClass',
            'OpusApp',
            'EXCEL7'
        ]
        
        # Check direct class matches
        if any(text_class in control_class for text_class in text_input_classes):
            return True
            
        if any(text_class in window_class for text_class in text_input_classes):
            return True
            
        # Additional check for browser text fields
        if any(container in window_class for container in container_classes):
            # Get cursor position
            cursor_pos = win32gui.GetCursorPos()
            # Convert screen coordinates to client coordinates
            client_pos = win32gui.ScreenToClient(foreground_window, cursor_pos)
            
            # Get window text at cursor position
            try:
                child_at_point = win32gui.ChildWindowFromPoint(foreground_window, client_pos)
                if child_at_point:
                    child_class = win32gui.GetClassName(child_at_point)
                    if any(text_class in child_class for text_class in text_input_classes):
                        return True
            except:
                pass
            
            # Additional check for browser input fields
            if 'Chrome' in window_class or 'Mozilla' in window_class:
                # Most browser text fields are detected by clicking
                # We'll consider it a text field if it's a potential browser input area
                return True
                
        return False
    except Exception as e:
        print(f"Error checking text field: {e}")
        return False

def get_voice_command():
    """Get voice command from microphone with simplified implementation"""
    global frame
    
    try:
        # Visual feedback - listening
        cv2.putText(frame, "Listening...", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        
        # Use microphone as source
        with sr.Microphone() as source:
            print("Listening...")
            
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            # Listen for audio
            audio = recognizer.listen(source)
            
            # Visual feedback - processing
            cv2.putText(frame, "Processing...", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Gesture Control", frame)
            cv2.waitKey(1)
            
            # Convert speech to text
            text = recognizer.recognize_google(audio)
            print(f"Recognized: {text}")
            
            # Visual feedback - recognized
            cv2.putText(frame, f"Recognized: {text}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Gesture Control", frame)
            cv2.waitKey(1)
            
            return text.lower()
            
    except sr.UnknownValueError:
        print("Could not understand audio")
        cv2.putText(frame, "Could not understand audio", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
        
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        cv2.putText(frame, "Speech recognition error", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""
        
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        cv2.putText(frame, "Error occurred", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        cv2.waitKey(1)
        return ""

def handle_text_input_mode(gesture_name, thumb_direction):
    """Handle gestures in text input mode"""
    global is_text_input_mode, frame
    
    if gesture_name == "✊ Fist":
        # Exit text input mode
        voice_command = get_voice_command()
        if voice_command == "quit":
            is_text_input_mode = False
            print("Exiting text input mode")
            return
    
    elif gesture_name == "🖐 Open Palm":
        # Move cursor left/right based on hand position
        if thumb_direction == "Up":
            pyautogui.press('left')
        else:
            pyautogui.press('right')
    
    elif gesture_name == "🤟 I Love You":
        # Backspace
        pyautogui.press('backspace')
    
    elif gesture_name == "👍 Thumbs Up":
        # Take voice input and type it
        print("Please speak your text...")
        voice_command = get_voice_command()
        if voice_command and voice_command != "quit":
            try:
                # Visual feedback - typing
                cv2.putText(frame, f"Typing: {voice_command}", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Gesture Control", frame)
                cv2.waitKey(1)
                
                # Ensure text box is focused
                pyautogui.click()
                time.sleep(0.1)  # Small delay to ensure focus
                
                # Type the recognized text
                pyautogui.write(voice_command)
                pyautogui.press('space')  # Add space after text
                
                print(f"Successfully typed: {voice_command}")
                
            except Exception as e:
                print(f"Error typing text: {e}")
                cv2.putText(frame, "Error typing text", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Gesture Control", frame)
                cv2.waitKey(1)

def perform_action(current_gesture, hand_landmarks, thumb_direction):
    """Perform actions based on gestures and transitions"""
    global last_gesture, last_gesture_time, last_action_time, last_hand_position, is_text_input_mode, last_text_input_check
    
    current_time = time.time()
    
    # Check for text input mode
    if current_time - last_text_input_check >= text_input_check_interval:
        current_in_text_field = is_cursor_in_text_field()
        
        # Enter text mode when clicking in text field
        if not is_text_input_mode and current_in_text_field:
            is_text_input_mode = True
            print("Entered text input mode")
            pyautogui.press('left')  # Ensure cursor is visible
        
        # Exit text mode when clicking outside text field
        elif is_text_input_mode and not current_in_text_field:
            is_text_input_mode = False
            print("Exited text input mode")
            
        last_text_input_check = current_time
    
    # Handle text input mode separately
    if is_text_input_mode:
        handle_text_input_mode(current_gesture, thumb_direction)
        return
    
    # Regular gesture control mode
    if current_gesture != last_gesture:
        last_gesture_time = current_time
        last_gesture = current_gesture
        last_hand_position = None
    elif current_time - last_gesture_time >= gesture_hold_time:
        # Perform actions based on gestures
        if current_gesture == "🖐 Open Palm":
            # Move mouse based on relative hand movement
            if last_hand_position is not None:
                # Calculate movement delta
                middle_tip = hand_landmarks.landmark[12]
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
            last_hand_position = (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y)
            
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

def draw_instructions(frame):
    """Draw instructions based on current mode"""
    if is_text_input_mode:
        instructions = [
            "Text Input Mode:",
            "👍 Thumbs Up: Voice Input",
            "🖐 Open Palm: Move Cursor (Up=Left, Down=Right)",
            "🤟 I Love You: Backspace",
            "✊ Fist + Say 'quit': Exit Text Mode",
            "Press 'q' to quit"
        ]
    else:
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

def get_primary_hand(hands_results):
    """Determine which hand should be the primary hand for gesture control"""
    global primary_hand_id, primary_hand_time, hand_tracking_history
    current_time = time.time()
    
    # Clean up old hands from history
    hand_tracking_history = {k: v for k, v in hand_tracking_history.items() 
                           if current_time - v < hand_tracking_timeout}
    
    # If no hands are detected, reset tracking
    if not hands_results.multi_hand_landmarks:
        primary_hand_id = None
        hand_tracking_history.clear()
        return None, -1
    
    # Update hand tracking history with new hands
    for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        # Calculate a unique identifier for this hand based on its initial position
        hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
        hand_id = f"hand_{idx}_{hand_center[0]:.2f}_{hand_center[1]:.2f}"
        
        # Record first appearance time if this is a new hand
        if hand_id not in hand_tracking_history:
            hand_tracking_history[hand_id] = current_time
    
    # If only one hand is present
    if len(hands_results.multi_hand_landmarks) == 1:
        primary_hand_id = 0
        return hands_results.multi_hand_landmarks[0], 0
    
    # If multiple hands are present, find the oldest hand
    if len(hands_results.multi_hand_landmarks) > 1:
        oldest_time = float('inf')
        oldest_idx = 0
        
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
            hand_id = f"hand_{idx}_{hand_center[0]:.2f}_{hand_center[1]:.2f}"
            
            if hand_id in hand_tracking_history:
                appear_time = hand_tracking_history[hand_id]
                if appear_time < oldest_time:
                    oldest_time = appear_time
                    oldest_idx = idx
        
        primary_hand_id = oldest_idx
        return hands_results.multi_hand_landmarks[oldest_idx], oldest_idx
    
    return None, -1

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Set up display window
cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1280, 720)

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
        # Get the primary hand for gesture control
        primary_hand, hand_id = get_primary_hand(hands_results)
        
        # Draw all detected hands but only process gestures for primary hand
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Calculate hand center for identification
            hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
            hand_id = f"hand_{idx}_{hand_center[0]:.2f}_{hand_center[1]:.2f}"
            
            # Only process gestures for the primary hand
            if idx == primary_hand_id:
                # Get enhanced gesture detection
                fingers, gesture_name, thumb_direction = detect_gesture(hand_landmarks)
                
                # Display primary hand status with time active
                time_active = time.time() - hand_tracking_history.get(hand_id, time.time())
                cv2.putText(frame, f"Primary Hand: {gesture_name} (Active: {time_active:.1f}s)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Perform action only for primary hand
                perform_action(gesture_name, hand_landmarks, thumb_direction)
            else:
                # Show inactive hand status with its age
                time_active = time.time() - hand_tracking_history.get(hand_id, time.time())
                cv2.putText(frame, f"Inactive Hand {idx} (Active: {time_active:.1f}s)", 
                           (10, 60 + (30 * idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # Display number of hands detected
        cv2.putText(frame, f"Hands Detected: {len(hands_results.multi_hand_landmarks)}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        # Display action status
        if time.time() - last_action_time < action_cooldown:
            cv2.putText(frame, "Action Cooldown...", (10, 180),
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
