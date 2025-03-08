from flask import Flask, render_template, request, redirect, url_for, jsonify
import json
import os

# Flask app initialization
app = Flask(__name__)

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
    "👍 Thumbs Up": "Start Voice Input",
    "👎 Thumbs Down": "Exit Text Input Mode"
}

# Load gesture actions from JSON file or use defaults
def load_gesture_actions():
    if os.path.exists('gesture_actions.json'):
        with open('gesture_actions.json', 'r', encoding='utf-8') as f:  # Specify encoding
            return json.load(f)
    return {
        "normal_mode": DEFAULT_NORMAL_MODE_ACTIONS,
        "text_input_mode": DEFAULT_TEXT_INPUT_MODE_ACTIONS
    }

# Save gesture actions to JSON file
def save_gesture_actions(actions):
    with open('gesture_actions.json', 'w', encoding='utf-8') as f:  # Specify encoding
        json.dump(actions, f, indent=4, ensure_ascii=False)  # Ensure non-ASCII characters are preserved

# Load initial actions
gesture_actions = load_gesture_actions()

# Define frequently used actions
frequent_actions = [
    'Move Mouse',
    'Update Hand Position',
    'Left Click',
    'Scroll Down',
    'Right Click',
    'Listen for Voice Command',
    'Move Cursor Left/Right',
    'Backspace',
    'Start Voice Input',
    'Exit Text Input Mode'
]

# Route for the main UI
@app.route('/')
def index():
    return render_template('index.html', actions=gesture_actions, frequent_actions=frequent_actions)

# Route to update gesture actions
@app.route('/update_action', methods=['POST'])
def update_action():
    mode = request.form.get('mode')  # Get the mode (normal_mode or text_input_mode)
    gesture = request.form.get('gesture')
    action = request.form.get('action')

    if mode in gesture_actions and gesture in gesture_actions[mode]:
        gesture_actions[mode][gesture] = action
        save_gesture_actions(gesture_actions)  # Save changes to JSON file
    return redirect(url_for('index'))

# Route to fetch current actions in JSON format
@app.route('/get_actions', methods=['GET'])
def get_actions():
    return jsonify(gesture_actions)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True,port=5001)