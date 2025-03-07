# actions.py
import pyautogui
import time
import cv2
from speech import get_voice_command
from utils import is_cursor_in_text_field

last_gesture_time = time.time()
last_action_time = time.time()
last_hand_position = None
last_i_love_you_time = 0  # New variable to throttle "I Love You" left clicks

def handle_text_input_mode(gesture_name, thumb_direction, frame, set_text_mode_callback):
    if gesture_name == "✊ Fist":
        voice_command = get_voice_command(frame)
        if voice_command == "quit":
            set_text_mode_callback(False)
            print("Exiting text input mode")
    elif gesture_name == "🖐 Open Palm":
        if thumb_direction == "Up":
            pyautogui.press('left')
        else:
            pyautogui.press('right')
    elif gesture_name == "🤟 I Love You":
        pyautogui.press('backspace')
    elif gesture_name == "👍 Thumbs Up":
        print("Please speak your text...")
        voice_command = get_voice_command(frame)
        if voice_command and voice_command != "quit":
            try:
                cv2.putText(frame, f"Typing: {voice_command}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Gesture Control", frame)
                cv2.waitKey(1)
                pyautogui.click()
                time.sleep(0.1)
                pyautogui.write(voice_command)
                pyautogui.press('space')
                print(f"Successfully typed: {voice_command}")
            except Exception as e:
                print(f"Error typing text: {e}")
                cv2.putText(frame, "Error typing text", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Gesture Control", frame)
                cv2.waitKey(1)

def perform_action(gesture_name, hand_landmarks, thumb_direction, frame, screen_width, screen_height):
    """Perform actions based on the detected gesture in normal mode"""
    global last_gesture_time, last_action_time, last_hand_position, last_i_love_you_time
    current_time = time.time()
    
    if is_cursor_in_text_field():
        pass  # Manage text mode state via callbacks in main.py if needed

    if gesture_name == "🖐 Open Palm":
        if last_hand_position is not None:
            middle_tip = hand_landmarks.landmark[12]
            delta_x = (middle_tip.x - last_hand_position[0]) * screen_width * 2.0
            delta_y = (middle_tip.y - last_hand_position[1]) * screen_height * 2.0
            current_x, current_y = pyautogui.position()
            new_x = current_x + int(delta_x * 0.5)
            new_y = current_y + int(delta_y * 0.5)
            new_x = max(0, min(new_x, screen_width - 1))
            new_y = max(0, min(new_y, screen_height - 1))
            pyautogui.moveTo(new_x, new_y)
        last_hand_position = (hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y)
    elif gesture_name == "🤟 I Love You":
        # Only trigger left click if 2 seconds have passed since the last click
        if current_time - last_i_love_you_time >= 2:
            pyautogui.click()
            last_i_love_you_time = current_time
            last_action_time = current_time
    elif gesture_name == "🤘 Yo":
        pyautogui.rightClick()
        last_action_time = current_time
    elif gesture_name == "✊ Fist":
        last_hand_position = None
    elif gesture_name == "👍 Thumbs Up":
        scroll_amount = 20 if thumb_direction == "Up" else -20
        pyautogui.scroll(scroll_amount)
        last_action_time = current_time
    elif gesture_name == "📞 Call Me":
        pyautogui.scroll(-20)
        last_action_time = current_time
