# main.py
import cv2
import time
import numpy as np
import pyautogui
from config import holistic, hands, is_text_input_mode, screen_width, screen_height, is_listening, last_action_time, action_cooldown, mp_drawing, mp_hands, mp_drawing_styles
from gesture import detect_gesture
from actions import perform_action
from utils import draw_instructions, get_primary_hand, is_cursor_in_text_field
from video import init_video, release_video
from face import is_authorized_face
from speech import start_voice_input

# Initialize video capture
cap = init_video()

# Variables for face recognition check
last_face_check_time = 0
face_check_interval = 3  # seconds
authorized = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally for mirror view
    frame = cv2.flip(frame, 1)

    # Periodically check for an authorized face
    current_time = time.time()
    if current_time - last_face_check_time > face_check_interval:
        authorized = is_authorized_face(frame)
        last_face_check_time = current_time

    # If face is not authorized, display a warning and skip gesture processing.
    if not authorized:
        cv2.putText(frame, "Unauthorized face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Gesture Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    # Create a semi-transparent overlay for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Draw instructions on the frame
    draw_instructions(frame, is_text_input_mode)

    if hands_results.multi_hand_landmarks:
        # Use only the primary hand.
        primary_hand, hand_idx = get_primary_hand(hands_results)
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            if idx == hand_idx:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                fingers, gesture_name, thumb_direction = detect_gesture(hand_landmarks)
                cv2.putText(frame, f"Primary Hand: {gesture_name}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Activate text mode if the gesture is "call me gesture" and the cursor is on a text box.
                if gesture_name == "call me gesture" and is_cursor_in_text_field():
                    if not is_text_input_mode:
                        from config import is_text_input_mode
                        is_text_input_mode = True
                        print("Text mode activated via call me gesture")
                        start_voice_input()
                    cv2.putText(frame, "Text Mode Activated", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Gesture Control", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                else:
                    perform_action(gesture_name, hand_landmarks, thumb_direction)
            else:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.putText(frame, f"Inactive Hand {idx}",
                            (10, 60 + (30 * idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        cv2.putText(frame, f"Hands Detected: {len(hands_results.multi_hand_landmarks)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if time.time() - last_action_time < action_cooldown:
            cv2.putText(frame, "Action Cooldown...", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if is_listening:
        cv2.putText(frame, "Listening for voice input...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_video(cap)
holistic.close()
hands.close()
