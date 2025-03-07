# utils.py
import cv2
import time
import win32gui
import numpy as np
from config import hand_tracking_history, hand_tracking_timeout, primary_hand_id

def is_cursor_in_text_field():
    """Enhanced check if cursor is in a text input field"""
    try:
        foreground_window = win32gui.GetForegroundWindow()
        window_title = win32gui.GetWindowText(foreground_window)
        window_class = win32gui.GetClassName(foreground_window)
        cursor_pos = win32gui.GetCursorPos()
        client_pos = win32gui.ScreenToClient(foreground_window, cursor_pos)
        
        if "WhatsApp" in window_title:
            window_rect = win32gui.GetWindowRect(foreground_window)
            left, top, right, bottom = window_rect
            window_height = bottom - top
            if client_pos[1] > window_height * 0.8:
                return True
                
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
            'WhatsAppWindowClass',
            'Chrome_WidgetWin_1'  # Added for Google Chrome search text boxes
        ]
        
        try:
            child_at_point = win32gui.ChildWindowFromPoint(foreground_window, client_pos)
            if child_at_point:
                child_class = win32gui.GetClassName(child_at_point)
                child_text = win32gui.GetWindowText(child_at_point)
                if ("WhatsApp" in window_title and 
                    (child_class in text_input_classes or 
                     "Type a message" in child_text or 
                     "message" in child_text.lower())):
                    return True
                    
                if any(text_class in child_class for text_class in text_input_classes):
                    return True
        except:
            pass
            
        return False
        
    except Exception as e:
        print(f"Error checking text field: {e}")
        return False

def draw_instructions(frame, is_text_input_mode):
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
    current_time = time.time()
    
    # Clean up old hands from history
    global hand_tracking_history
    hand_tracking_history = {k: v for k, v in hand_tracking_history.items() 
                             if current_time - v < hand_tracking_timeout}
    
    if not hands_results.multi_hand_landmarks:
        global primary_hand_id
        primary_hand_id = None
        hand_tracking_history.clear()
        return None, -1
    
    for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
        hand_id = f"hand_{idx}{hand_center[0]:.2f}{hand_center[1]:.2f}"
        if hand_id not in hand_tracking_history:
            hand_tracking_history[hand_id] = current_time
    
    if len(hands_results.multi_hand_landmarks) == 1:
        primary_hand_id = 0
        return hands_results.multi_hand_landmarks[0], 0
    
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
