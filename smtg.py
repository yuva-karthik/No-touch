# main.py
import cv2
import time
import numpy as np
import pyautogui
from config import holistic, hands, is_text_input_mode, screen_width, screen_height
from gesture import detect_gesture
from actions import perform_action
from utils import draw_instructions, get_primary_hand
from video import init_video, release_video

# Initialize video capture
cap = init_video()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe holistic and hands models
    holistic_results = holistic.process(rgb_frame)
    hands_results = hands.process(rgb_frame)
    
    # Create a semi-transparent overlay for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 400), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Draw instructions on the frame
    draw_instructions(frame, is_text_input_mode)
    
    if hands_results.multi_hand_landmarks:
        primary_hand, hand_idx = get_primary_hand(hands_results)
        
        # Draw all detected hands and process primary hand for gestures
        for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
            from config import mp_drawing, mp_hands, mp_drawing_styles
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
            hand_id = f"hand_{idx}{hand_center[0]:.2f}{hand_center[1]:.2f}"
            
            if idx == hand_idx:
                # For the primary hand, detect gesture and perform its action
                fingers, gesture_name, thumb_direction = detect_gesture(hand_landmarks)
                cv2.putText(frame, f"Primary Hand: {gesture_name}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                perform_action(gesture_name, hand_landmarks, thumb_direction)
            else:
                cv2.putText(frame, f"Inactive Hand {idx}", 
                            (10, 60 + (30 * idx)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        cv2.putText(frame, f"Hands Detected: {len(hands_results.multi_hand_landmarks)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        from config import last_action_time, action_cooldown
        if time.time() - last_action_time < action_cooldown:
            cv2.putText(frame, "Action Cooldown...", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    from config import is_listening
    if is_listening:
        cv2.putText(frame, "Listening for voice input...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Gesture Control", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

release_video(cap)
holistic.close()
hands.close()
