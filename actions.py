import json
import time
import cv2
import pyautogui
from speech import get_voice_command, start_voice_input
from utils import is_cursor_in_text_field
import config

# Default gesture-action mappings for Normal Mode
DEFAULT_NORMAL_MODE_ACTIONS = {
    "🖐 Open Palm": "Move Mouse",
    "✊ Fist": "Update Hand Position",
    "🤟 I Love You": "Left Click",
    "👍 Thumbs Up": "Scroll Down",
    "🤘 Yo": "Right Click",
    "👎 Thumbs Down": "Exit Text Input Mode"
}

# Default gesture-action mappings for Text Input Mode
DEFAULT_TEXT_INPUT_MODE_ACTIONS = {
    "✊ Fist": "Listen for Voice Command",
    "🖐 Open Palm": "Move Cursor Left/Right",
    "🤟 I Love You": "Backspace",
    "👍 Thumbs Up": "Backspace",
    "👎 Thumbs Down": "Exit Text Input Mode"
}

# Load gesture-action mappings from JSON file or use defaults
def load_gesture_actions():
    try:
        with open("gesture_actions.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("gesture_actions.json not found. Using default mappings.")
        return {
            "normal_mode": DEFAULT_NORMAL_MODE_ACTIONS,
            "text_input_mode": DEFAULT_TEXT_INPUT_MODE_ACTIONS
        }
    except json.JSONDecodeError:
        print("gesture_actions.json is invalid. Using default mappings.")
        return {
            "normal_mode": DEFAULT_NORMAL_MODE_ACTIONS,
            "text_input_mode": DEFAULT_TEXT_INPUT_MODE_ACTIONS
        }

# Perform actions based on the current mode and gesture
def perform_action(current_gesture, hand_landmarks, thumb_direction):
    global frame  # 'frame' is assumed to be set in the main loop

    # Reload gesture-action mappings dynamically
    gesture_actions = load_gesture_actions()

    # Determine the current mode
    mode_actions = gesture_actions["text_input_mode"] if config.is_text_input_mode else gesture_actions["normal_mode"]

    if current_gesture in mode_actions:
        action = mode_actions[current_gesture]

        if action == "Move Mouse":
            middle_tip = hand_landmarks.landmark[12]
            current_hand = (middle_tip.x, middle_tip.y)

            if config.last_gesture != "🖐 Open Palm" or config.last_hand_position is None:
                config.last_hand_position = current_hand
            else:
                delta_x = (current_hand[0] - config.last_hand_position[0]) * config.screen_width * config.MOUSE_SENSITIVITY
                delta_y = (current_hand[1] - config.last_hand_position[1]) * config.screen_height * config.MOUSE_SENSITIVITY

                current_x, current_y = pyautogui.position()
                new_x = current_x + int(delta_x)
                new_y = current_y + int(delta_y)

                new_x = max(0, min(new_x, config.screen_width - 1))
                new_y = max(0, min(new_y, config.screen_height - 1))

                pyautogui.moveTo(new_x, new_y)

            config.last_hand_position = current_hand

            # Additional behavior for thumb direction
            if thumb_direction == "Up":
                pyautogui.press('left')
            elif thumb_direction == "Down":
                pyautogui.press('right')

        elif action == "Update Hand Position":
            middle_tip = hand_landmarks.landmark[12]
            config.last_hand_position = (middle_tip.x, middle_tip.y)

            # Additional behavior in text input mode
            if config.is_text_input_mode:
                voice_command = get_voice_command()
                if voice_command == "quit":
                    config.is_text_input_mode = False
                    print("Exited text input mode")

        elif action == "Left Click":
            if time.time() - config.last_action_time >= config.action_cooldown:
                pyautogui.click()
                config.last_action_time = time.time()
                print("Left Click performed")

                # Additional behavior for text input mode
                if is_cursor_in_text_field():
                    if not config.is_text_input_mode:
                        config.is_text_input_mode = True
                        print("Entered text input mode")
                        start_voice_input()
                else:
                    if config.is_text_input_mode:
                        config.is_text_input_mode = False
                        print("Exited text input mode")
            else:
                remaining_cooldown = config.action_cooldown - (time.time() - config.last_action_time)
                print(f"Click cooldown: {remaining_cooldown:.1f}s")

        elif action == "Scroll Down":
            pyautogui.scroll(-100)
            print("Scrolled Down")

        elif action == "Right Click":
            pyautogui.rightClick()
            print("Right Click performed")

        elif action == "Listen for Voice Command":
            if config.is_text_input_mode:
                voice_command = get_voice_command()
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

        elif action == "Backspace":
            if config.is_text_input_mode:
                pyautogui.press('backspace')
                print("Backspace pressed")

        elif action == "Start Voice Input":
            if config.is_text_input_mode and not config.is_listening:
                print("Starting voice input...")
                start_voice_input()

        elif action == "Exit Text Input Mode":
            if config.is_text_input_mode:
                config.is_text_input_mode = False
                print("Exited text input mode")

    config.last_gesture = current_gesture