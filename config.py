# config.py
import pyautogui
import time

# PyAutoGUI settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

# Text input mode settings
IS_TEXT_INPUT_MODE = False
LAST_TEXT_INPUT_CHECK = time.time()
TEXT_INPUT_CHECK_INTERVAL = 0.5  # seconds

# Mediapipe configuration
MIN_DETECTION_CONFIDENCE = 0.3
MIN_TRACKING_CONFIDENCE = 0.3

# Gesture tracking settings
GESTURE_HOLD_TIME = 0.3
ACTION_COOLDOWN = 0.2

# Mouse control settings
MOUSE_SENSITIVITY = 2.0
SMOOTH_FACTOR = 0.5

# config.py

# Mini player settings
MINI_PLAYER_WIDTH = 320
MINI_PLAYER_HEIGHT = 240  # Updated from 180 to 240 for live camera feed
MINI_PLAYER_MARGIN = 20

# Video source: change from an mp4 file path to the default webcam (live feed)
VIDEO_PATH = 0
