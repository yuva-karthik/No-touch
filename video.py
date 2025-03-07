# video.py
import cv2
import numpy as np
import pyautogui
from config import MINI_PLAYER_WIDTH, MINI_PLAYER_HEIGHT

def resize_frame_with_aspect_ratio(frame, width=None, height=None):
    (h, w) = frame.shape[:2]
    if width is None and height is None:
        return frame
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

def display_mini_player(video_player, screen_width, screen_height):
    ret, video_frame = video_player.read()
    if ret:
        video_frame = cv2.flip(video_frame, 1)
        video_frame = resize_frame_with_aspect_ratio(video_frame, width=MINI_PLAYER_WIDTH)
        padding = np.zeros((MINI_PLAYER_HEIGHT + 40, MINI_PLAYER_WIDTH + 4, 3), dtype=np.uint8)
        cv2.putText(padding, "Mini Player", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        h, w = video_frame.shape[:2]
        y_offset = 35
        x_offset = 2
        padding[y_offset:y_offset+h, x_offset:x_offset+w] = video_frame
        cv2.imshow("Mini Player", padding)
        cv2.moveWindow("Mini Player",
                       screen_width - MINI_PLAYER_WIDTH - 50,
                       screen_height - MINI_PLAYER_HEIGHT - 100)
    else:
        video_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
