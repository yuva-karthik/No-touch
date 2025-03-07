# utils.py
import cv2
import win32gui

def draw_instructions(frame, is_text_input_mode):
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

def is_cursor_in_text_field():
    try:
        foreground_window = win32gui.GetForegroundWindow()
        focused_control = win32gui.GetFocus()
        window_class = win32gui.GetClassName(foreground_window)
        control_class = win32gui.GetClassName(focused_control) if focused_control else ""
        text_input_classes = [
            'Edit', 'RichEdit', 'RichEdit20W', 'RichEdit20A', 'TextBox',
            'RICHEDIT50W', 'Chrome_RenderWidgetHostHWND', 'MozillaWindowClass',
            'Scintilla', 'Notepad', 'WordPadClass', 'EXCEL7', 'OpusApp', 'SUMATRA_PDF_FRAME'
        ]
        container_classes = ['Chrome_WidgetWin_1', 'MozillaWindowClass', 'Notepad', 'WordPadClass', 'OpusApp', 'EXCEL7']
        if any(text_class in control_class for text_class in text_input_classes):
            return True
        if any(text_class in window_class for text_class in text_input_classes):
            return True
        if any(container in window_class for container in container_classes):
            return True
        return False
    except Exception as e:
        print(f"Error checking text field: {e}")
        return False
