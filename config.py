# config.py
import cv2
import mediapipe as mp
import pyautogui
import time
import speech_recognition as sr

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# PyAutoGUI configuration
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1

# Speech recognition initialization
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

# MediaPipe configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)

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

# Voice input global variable
is_listening = False
voice_input_thread = None
