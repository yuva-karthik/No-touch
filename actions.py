# actions.py
import time
import cv2
import pyautogui
from speech import get_voice_command, start_voice_input
from utils import is_cursor_in_text_field
import config

def handle_text_input_mode(gesture_name, thumb_direction):
    """Handle gestures in text input mode"""
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
        pyautogui.press('backspace')
    
    elif gesture_name == "👍 Thumbs Up":
        print("Please speak your text...")
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

def perform_action(current_gesture, hand_landmarks, thumb_direction):
    """Perform actions based on gestures and transitions"""
    import time
    import pyautogui
    import config  # Make sure to import the config module
    current_time = time.time()
    
    if current_gesture == "🖐 Open Palm":
        middle_tip = hand_landmarks.landmark[12]
        current_hand = (middle_tip.x, middle_tip.y)
        
        # If coming from a different gesture or no reference exists, update last_hand_position without moving
        if config.last_gesture != "🖐 Open Palm" or config.last_hand_position is None:
            config.last_hand_position = current_hand
        else:
            # Calculate movement delta based on continuous open palm movement
            delta_x = (current_hand[0] - config.last_hand_position[0]) * config.screen_width * config.MOUSE_SENSITIVITY
            delta_y = (current_hand[1] - config.last_hand_position[1]) * config.screen_height * config.MOUSE_SENSITIVITY
            
            # Get current mouse position
            current_x, current_y = pyautogui.position()
            new_x = current_x + int(delta_x)
            new_y = current_y + int(delta_y)
            
            # Ensure new coordinates are within screen bounds
            new_x = max(0, min(new_x, config.screen_width - 1))
            new_y = max(0, min(new_y, config.screen_height - 1))
            
            pyautogui.moveTo(new_x, new_y)
        
        # Update the reference hand position for the next frame
        config.last_hand_position = current_hand
        
    elif current_gesture == "✊ Fist":
        # When making a fist, update last_hand_position without moving the cursor
        middle_tip = hand_landmarks.landmark[12]
        config.last_hand_position = (middle_tip.x, middle_tip.y)
    
    elif current_gesture == "🤟 I Love You":
        if current_time - config.last_action_time >= config.action_cooldown:
            pyautogui.click()
            config.last_action_time = current_time
            print("Left click performed")
            from utils import is_cursor_in_text_field
            if is_cursor_in_text_field():
                if not config.is_text_input_mode:
                    config.is_text_input_mode = True
                    print("Entered text input mode")
                    from speech import start_voice_input
                    start_voice_input()
            else:
                if config.is_text_input_mode:
                    config.is_text_input_mode = False
                    print("Exited text input mode")
        else:
            remaining_cooldown = config.action_cooldown - (current_time - config.last_action_time)
            print(f"Click cooldown: {remaining_cooldown:.1f}s")
    
    elif current_gesture == "👍 Thumbs Up" and config.is_text_input_mode and not config.is_listening:
        print("Thumbs Up detected - Starting new voice input")
        from speech import start_voice_input
        start_voice_input()
    
    elif current_gesture == "🤘 Yo":
        pyautogui.rightClick()
    
    elif current_gesture == "👎 Thumbs Down":
        if config.is_text_input_mode:
            config.is_text_input_mode = False
            print("Exited text input mode")
    
    config.last_gesture = current_gesture
