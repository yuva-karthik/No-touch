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
import threading

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Configure PyAutoGUI
pyautogui.FAILSAFE = False
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
action_cooldown = 1.0  # Changed from 0.2 to 1.0 second
last_action_time = time.time()
primary_hand_id = None  # Track the ID of the primary hand
primary_hand_time = 0  # Time when primary hand was last active
hand_tracking_history = {}  # Track when each hand first appeared
hand_tracking_timeout = 2.0  # Time before removing hand from history

# Mouse control settings
MOUSE_SENSITIVITY = 2.0  # Adjust this value to change mouse movement sensitivity
last_hand_position = None
smooth_factor = 0.5  # Smoothing factor for mouse movement (0 to 1)

# Add this to your global variables at the top
is_listening = False
voice_input_thread = None

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
        (0, 0, 0, 0, 0): "✊ Fist",
        (1, 1, 1, 1, 1): "🖐 Open Palm",
        (1, 1, 1, 0, 0): "🤟 I Love You",
        (0, 1, 1, 1, 0): "🤘 Yo",
        (0, 0, 0, 0, 1): "👎 Thumbs Down"
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

def is_cursor_in_text_field():
    """Enhanced check if cursor is in a text input field"""
    try:
        # Get the foreground window and focused control
        foreground_window = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(foreground_window)
        window_class = win32gui.GetClassName(foreground_window)
        
        # Get cursor position
        cursor_pos = win32gui.GetCursorPos()
        client_pos = win32gui.ScreenToClient(foreground_window, cursor_pos)
        
        # Special handling for WhatsApp Desktop App
        if "WhatsApp" in window_title:
            # Get the screen coordinates of the window
            window_rect = win32gui.GetWindowRect(foreground_window)
            # Convert window rect to client coordinates
            left, top, right, bottom = window_rect
            window_height = bottom - top
            
            # WhatsApp text input is typically at the bottom of the window
            # Check if cursor is in the bottom portion of the window
            if client_pos[1] > window_height * 0.8:  # Bottom 20% of window
                return True
                
        # Regular text input checks
        text_input_classes = [
            'Edit',
            'RichEdit',
            'RichEdit20W',
            'RichEdit20A',
            'TextBox',
            'RICHEDIT50W',
            'Chrome_RenderWidgetHostHWND',
            'MozillaWindowClass',
            'Scintilla',
            'Notepad',
            'WordPadClass',
            'EXCEL7',
            'OpusApp',
            'WhatsAppWindowClass'
        ]
        
        # Try to get the control at cursor position
        try:
            child_at_point = win32gui.ChildWindowFromPoint(foreground_window, client_pos)
            if child_at_point:
                child_class = win32gui.GetClassName(child_at_point)
                child_text = win32gui.GetWindowText(child_at_point)
                
                # Check for WhatsApp specific elements
                if ("WhatsApp" in window_title and 
                    (child_class in text_input_classes or 
                     "Type a message" in child_text or 
                     "message" in child_text.lower())):
                    return True
                    
                # Check other text input classes
                if any(text_class in child_class for text_class in text_input_classes):
                    return True
        except:
            pass
            
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

def start_voice_input():
    """Start voice input in a separate thread"""
    global is_listening, voice_input_thread
    
    def voice_input_worker():
        global is_listening
        try:
            with sr.Microphone() as source:
                print("Listening...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
                
            try:
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                
                # Type the recognized text
                pyautogui.write(text)
                pyautogui.press('space')
                print(f"Successfully typed: {text}")
                
            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                
        except Exception as e:
            print(f"Error in voice input: {e}")
        finally:
            is_listening = False
    
    if not is_listening:
        is_listening = True
        voice_input_thread = threading.Thread(target=voice_input_worker)
        voice_input_thread.daemon = True
        voice_input_thread.start()

def perform_action(current_gesture, hand_landmarks, thumb_direction):
    """Perform actions based on gestures and transitions"""
    global last_gesture, last_gesture_time, last_action_time, last_hand_position, is_text_input_mode
    global screen_width, screen_height, is_listening
    
    current_time = time.time()
    
    # Always handle cursor movement regardless of mode
    if current_gesture == "🖐 Open Palm":
        # Get current hand position
        middle_tip = hand_landmarks.landmark[12]
        current_hand = (middle_tip.x, middle_tip.y)
        
        # If coming from a different gesture, just update hand position without moving
        if last_gesture != "🖐 Open Palm":
            last_hand_position = current_hand
        # Only move if we have a previous hand position to compare with
        elif last_hand_position is not None:
            # Calculate movement delta
            delta_x = (current_hand[0] - last_hand_position[0]) * screen_width * MOUSE_SENSITIVITY
            delta_y = (current_hand[1] - last_hand_position[1]) * screen_height * MOUSE_SENSITIVITY
            
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            
            # Calculate new position based on hand movement
            new_x = current_x + int(delta_x)
            new_y = current_y + int(delta_y)
            
            # Ensure coordinates stay within screen bounds
            new_x = max(0, min(new_x, screen_width - 1))
            new_y = max(0, min(new_y, screen_height - 1))
            
            # Move mouse
            pyautogui.moveTo(new_x, new_y)
        
        # Update last hand position
        last_hand_position = current_hand
        
    elif current_gesture == "✊ Fist":
        # When making a fist, just update the hand position without moving cursor
        middle_tip = hand_landmarks.landmark[12]
        last_hand_position = (middle_tip.x, middle_tip.y)
    
    elif current_gesture == "🤟 I Love You":
        # Left click with cooldown
        if current_time - last_action_time >= action_cooldown:
            pyautogui.click()
            last_action_time = current_time
            print("Left click performed")
            
            # Check if clicked in text field
            if is_cursor_in_text_field():
                if not is_text_input_mode:
                    is_text_input_mode = True
                    print("Entered text input mode")
                    # Start initial voice input
                    start_voice_input()
            else:
                if is_text_input_mode:
                    is_text_input_mode = False
                    print("Exited text input mode")
        else:
            remaining_cooldown = action_cooldown - (current_time - last_action_time)
            print(f"Click cooldown: {remaining_cooldown:.1f}s")
    
    elif current_gesture == "👍 Thumbs Up" and is_text_input_mode and not is_listening:
        # Start new voice input when Thumbs Up is detected in text input mode
        print("Thumbs Up detected - Starting new voice input")
        start_voice_input()
    
    elif current_gesture == "🤘 Yo":
        # Right click
        pyautogui.rightClick()
    
    elif current_gesture == "👎 Thumbs Down":
        if is_text_input_mode:
            is_text_input_mode = False
            print("Exited text input mode")
    
    # Update last gesture
    last_gesture = current_gesture

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
        hand_id = f"hand_{idx}{hand_center[0]:.2f}{hand_center[1]:.2f}"
        
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
            hand_id = f"hand_{idx}{hand_center[0]:.2f}{hand_center[1]:.2f}"
            
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

def capture_reference_photo(photo_path):
    """Load a reference photo for face authentication from a specified path."""
    if os.path.exists(photo_path):
        print(f"Loading reference photo from: {photo_path}")
        reference_photo = photo_path  # Store the path for later use
    else:
        print("Error: The specified photo path does not exist.")
        return False
    return True

def run():
    # Ask user for the path to the reference photo
    photo_path = input("Enter the path to the reference photo: ")
    if not capture_reference_photo(photo_path):
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save the current frame for face authentication
        cv2.imwrite("current_frame.jpg", frame)  # Save the current frame
        
        # Authenticate the face using the loaded reference photo
        if not authenticate_face():
            print("Face not recognized. Exiting application.")
            break  # Exit if face is not recognized
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Process frame for hand detection
        holistic_results, hands_results = self.gesture_recognizer.process_frame(frame)
        
        # Create a semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw instructions
        draw_instructions(frame)
        
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
                hand_id = f"hand_{idx}{hand_center[0]:.2f}{hand_center[1]:.2f}"
                
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
        
        # Add voice input status
        if is_listening:
            cv2.putText(frame, "Listening for voice input...", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display the main frame
        cv2.imshow("Gesture Control", frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    hands.close()

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
            hand_id = f"hand_{idx}{hand_center[0]:.2f}{hand_center[1]:.2f}"
            
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
    
    # Add voice input status
    if is_listening:
        cv2.putText(frame, "Listening for voice input...", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display the main frame
    cv2.imshow("Gesture Control", frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
holistic.close()
hands.close()