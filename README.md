# Gesture Control Application

A Python application that enables hands-free computer control using gestures and voice commands. The application uses computer vision to detect hand gestures and provides intuitive control over mouse movement, clicking, and text input.

## Features

- Cursor control using hand gestures
- Left and right click functionality
- Text input mode with voice recognition
- Smooth cursor movement with position smoothing
- Click cooldown to prevent accidental clicks
- Visual feedback for all actions
- Multiple ways to exit text input mode

## Gestures

- **Open Palm**: Move the cursor
- **I Love You**: Left click (with 2-second cooldown)
- **Thumbs Up**: Start voice input (in text mode)
- **Thumbs Down**: Exit text input mode
- **Yo**: Right click (only in gesture mode)
- **Fist + "quit"**: Exit application

## Requirements

- Python 3.8 or higher
- Webcam
- Microphone (for voice input)
- The packages listed in `requirements.txt`

## Installation

1. Clone the repository
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python myapp.py
   ```

2. The application will open your webcam and start tracking your hand gestures.

3. Use the gestures listed above to control your computer.

4. Press 'q' to quit the application.

## Text Input Mode

- Enters automatically when clicking on a text field
- Voice input starts when making a Thumbs Up gesture
- Exit using:
  - Thumbs Down gesture
  - 'e' key
  - Making a fist and saying "quit"

## Modules

- `hand_tracking.py`: Handles hand detection and tracking
- `gesture_recognition.py`: Recognizes hand gestures
- `cursor_control.py`: Controls mouse movement and clicks
- `voice_input.py`: Manages voice recognition
- `ui_renderer.py`: Renders UI elements and visual feedback

## Notes

- The application uses PyAutoGUI's failsafe is disabled. Be careful when using the application.
- Voice recognition requires an internet connection (uses Google Speech Recognition).
- Click cooldown is set to 2 seconds to prevent accidental clicks.
- The application works best in good lighting conditions.

## Troubleshooting

1. If the cursor movement is too sensitive:
   - Adjust the `smoothing_factor` in `CursorController`

2. If voice recognition is not working:
   - Check your microphone settings
   - Ensure you have an active internet connection
   - Try adjusting your microphone volume

3. If hand detection is unreliable:
   - Ensure good lighting
   - Keep your hand within the camera frame
   - Try to maintain a clear background

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements. 