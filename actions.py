# actions.py
import time
import cv2
import pyautogui
from speech import get_voice_command, start_voice_input
from utils import is_cursor_in_text_field
import config

def handle_text_input_mode(gesture_name, thumb_direction):
    """Handle gestures in text input mode."""
    global frame  # 'frame' is assumed to be set in main loop
    if gesture_name == "✊ Fist":
        voice_command = get_voice_command()
        if voice_command == "quit":
            config.is_text_input_mode = False
            print("Exiting text input mode")
            return

    elif gesture_name == "🖐 Open Palm":
        if thumb_direction == "Up":
            pyautogui.press('left')
        else:
            pyautogui.press('right')

    elif gesture_name == "🤟 I Love You":
        # Perform a left click.
        pyautogui.click()
        print("Left click performed")

    elif gesture_name == "👍 Thumbs Up":
        # Single-hand thumbs up always scrolls up.
        pyautogui.scroll(50)
        print("Scrolling up")

    elif gesture_name in ["👎 Thumbs Down", "👍⬇ Thumbs Up Down"]:
        # Treat both variants as scrolling down.
        pyautogui.scroll(-50)
        print("Scrolling down")

def perform_action(current_gesture, hand_landmarks, thumb_direction):
    """Perform actions based on gestures and transitions."""
    current_time = time.time()

    # New branch for call me gesture: Activate text mode and start voice input.
    if current_gesture == "call me gesture":
        if is_cursor_in_text_field():
            if not config.is_text_input_mode:
                config.is_text_input_mode = True
                print("Text mode activated via call me gesture")
                start_voice_input()
        else:
            print("Call me gesture detected but cursor not in text field.")
        # Exit after handling the call me gesture.
        config.last_gesture = current_gesture
        return

    if current_gesture == "🖐 Open Palm":
        middle_tip = hand_landmarks.landmark[12]
        current_hand = (middle_tip.x, middle_tip.y)
        if config.last_gesture != "🖐 Open Palm" or config.last_hand_position is None:
            config.last_hand_position = current_hand
        else:
            # Calculate delta for smooth cursor movement.
            delta_x = (current_hand[0] - config.last_hand_position[0]) * config.screen_width * config.MOUSE_SENSITIVITY
            delta_y = (current_hand[1] - config.last_hand_position[1]) * config.screen_height * config.MOUSE_SENSITIVITY
            current_x, current_y = pyautogui.position()
            new_x = current_x + int(delta_x)
            new_y = current_y + int(delta_y)
            new_x = max(0, min(new_x, config.screen_width - 1))
            new_y = max(0, min(new_y, config.screen_height - 1))
            pyautogui.moveTo(new_x, new_y)
        config.last_hand_position = current_hand

    elif current_gesture == "✊ Fist":
        middle_tip = hand_landmarks.landmark[12]
        config.last_hand_position = (middle_tip.x, middle_tip.y)

    elif current_gesture == "🤟 I Love You":
        if current_time - config.last_action_time >= config.action_cooldown:
            pyautogui.click()
            config.last_action_time = current_time
            print("Left click performed")
        else:
            remaining_cooldown = config.action_cooldown - (current_time - config.last_action_time)
            print(f"Click cooldown: {remaining_cooldown:.1f}s")

    elif current_gesture == "👍 Thumbs Up":
        pyautogui.scroll(50)
        print("Scrolling up")

    elif current_gesture in ["👎 Thumbs Down", "👍⬇ Thumbs Up Down"]:
        pyautogui.scroll(-50)
        print("Scrolling down")

    elif current_gesture == "🤘 Yo":
        pyautogui.rightClick()

    config.last_gesture = current_gesture
