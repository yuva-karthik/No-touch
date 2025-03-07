# gesture.py
import numpy as np
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_holistic = mp.solutions.holistic

def calculate_finger_angles(landmarks):
    angles = []
    finger_joints = [
        [2, 3, 4],    # Thumb
        [5, 6, 8],    # Index
        [9, 10, 12],  # Middle
        [13, 14, 16], # Ring
        [17, 18, 20]  # Pinky
    ]
    for joint in finger_joints:
        p1 = np.array([landmarks[joint[0]].x, landmarks[joint[0]].y])
        p2 = np.array([landmarks[joint[1]].x, landmarks[joint[1]].y])
        p3 = np.array([landmarks[joint[2]].x, landmarks[joint[2]].y])
        v1 = p1 - p2
        v2 = p3 - p2
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angles.append(np.degrees(angle))
    return angles

def get_extended_fingers(landmarks):
    angles = calculate_finger_angles(landmarks)
    thresholds = [130, 140, 140, 140, 140]
    return [1 if angle > threshold else 0 for angle, threshold in zip(angles, thresholds)]

def get_gesture_name(fingers, angles):
    gesture_dict = {
        (0, 0, 0, 0, 0): "✊ Fist",
        (1, 1, 1, 1, 1): "🖐 Open Palm",
        (1, 1, 1, 0, 0): "🤟 I Love You",
        (0, 1, 1, 1, 0): "🤘 Yo",
        (1, 0, 0, 0, 0): "👍 Thumbs Up",
        (0, 0, 0, 0, 1): "👎 Thumbs Down",
        (1, 0, 0, 0, 1): "📞 Call Me"
    }
    gesture = gesture_dict.get(tuple(fingers), "Analyzing...")
    if gesture in ["👍 Thumbs Up", "👎 Thumbs Down"] and angles[0] < 100:
        return "Analyzing..."
    return gesture

def detect_gesture(hand_landmarks, perform_action_callback):
    landmarks_list = [lm for lm in hand_landmarks.landmark]
    fingers = get_extended_fingers(landmarks_list)
    angles = calculate_finger_angles(landmarks_list)
    gesture_name = get_gesture_name(fingers, angles)
    thumb_tip = landmarks_list[4]
    thumb_base = landmarks_list[2]
    thumb_direction = "Up" if thumb_tip.y < thumb_base.y else "Down"
    perform_action_callback(gesture_name, hand_landmarks, thumb_direction)
    return fingers, gesture_name, thumb_direction

def get_primary_hand(hands_results, hand_tracking_history, hand_tracking_timeout):
    current_time = time.time()
    hand_tracking_history = {k: v for k, v in hand_tracking_history.items() 
                             if current_time - v < hand_tracking_timeout}
    if not hands_results.multi_hand_landmarks:
        hand_tracking_history.clear()
        return None, -1
    for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
        hand_id = f"hand_{idx}_{hand_center[0]:.2f}_{hand_center[1]:.2f}"
        if hand_id not in hand_tracking_history:
            hand_tracking_history[hand_id] = current_time
    if len(hands_results.multi_hand_landmarks) == 1:
        return hands_results.multi_hand_landmarks[0], 0
    oldest_time = float('inf')
    oldest_idx = 0
    for idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        hand_center = np.mean([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
        hand_id = f"hand_{idx}_{hand_center[0]:.2f}_{hand_center[1]:.2f}"
        appear_time = hand_tracking_history.get(hand_id, current_time)
        if appear_time < oldest_time:
            oldest_time = appear_time
            oldest_idx = idx
    return hands_results.multi_hand_landmarks[oldest_idx], oldest_idx
