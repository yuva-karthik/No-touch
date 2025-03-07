# main.py
import cv2
import pyautogui
import time
import numpy as np
import mediapipe as mp

from config import *
from gesture import detect_gesture, get_primary_hand
from speech import get_voice_command
from video import display_mini_player
from actions import perform_action, handle_text_input_mode
from utils import draw_instructions

pyautogui.FAILSAFE = False  # Disable the fail-safe

mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
holistic = mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                                min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# Initialize webcam for main feed
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Use the live camera feed for the mini player as well
video_player = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_DSHOW)
video_loaded = video_player.isOpened()

cv2.namedWindow("Gesture Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gesture Control", 1280, 720)
cv2.namedWindow("Mini Player", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mini Player", MINI_PLAYER_WIDTH, MINI_PLAYER_HEIGHT)
screen_width, screen_height = pyautogui.size()

hand_tracking_history = {}
HAND_TRACKING_TIMEOUT = 2.0
primary_hand_id = None
is_text_input_mode = IS_TEXT_INPUT_MODE

def set_text_mode(state):
    global is_text_input_mode
    is_text_input_mode = state

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    draw_instructions(frame, is_text_input_mode)
    
    if video_loaded:
        display_mini_player(video_player, screen_width, screen_height)
    
    if hands_results.multi_hand_landmarks:
        primary_hand, primary_hand_id = get_primary_hand(hands_results, hand_tracking_history, HAND_TRACKING_TIMEOUT)
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            if idx == primary_hand_id:
                _, gesture_name, thumb_direction = detect_gesture(
                    hand_landmarks,
                    lambda g, hl, td: perform_action(g, hl, td, frame, screen_width, screen_height)
                )
                cv2.putText(frame, f"Primary: {gesture_name}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_loaded:
    video_player.release()
cv2.destroyAllWindows()
holistic.close()
hands.close()
