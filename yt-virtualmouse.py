# import cv2
# import mediapipe as mp
# import pyautogui
# from pynput.mouse import Button, Controller
# import random
# import util

# mouse = Controller()

# screen_width, screen_height = pyautogui.size()

# mpHands = mp.solutions.hands
# hands = mpHands.Hands(
#     static_image_mode = False,
#     model_complexity = 1,
#     min_detection_confidence = 0.7,
#     min_tracking_confidence = 0.7,
#     max_num_hands = 1
# )

# def find_finger_tip(processed):
#     if processed.multi_hand_landmarks:
#         hand_landmarks = processed.multi_hand_landmarks[0]
#         return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    
#     return None

# def move_mouse(index_finger_tip):
#     if index_finger_tip is not None:
#         x = int(index_finger_tip.x * screen_width)
#         y = int(index_finger_tip.y * screen_height)
#         pyautogui.moveTo(x,y)

# def is_left_click(landmarks_list, thumb_index_dist):
#     return (util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
#             util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) > 90 and
#             thumb_index_dist > 50
#             )

# def is_right_click(landmarks_list, thumb_index_dist):
#     return (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
#             util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 and
#             thumb_index_dist > 50
#             )

# def is_double_click(landmarks_list, thumb_index_dist):
#     return (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
#             util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
#             thumb_index_dist > 50
#             )

# def is_screenshot(landmarks_list, thumb_index_dist):
#     return (util.get_angle(landmarks_list[9], landmarks_list[10], landmarks_list[12]) < 50 and
#             util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) < 50 and
#             thumb_index_dist < 50
#             )

# def detect_gestures(frame, landmarks_list, processed):
#     if len(landmarks_list) >= 21:
        
#         index_finger_tip = find_finger_tip(processed)
#         thumb_index_dist = util.get_distance([landmarks_list[4], landmarks_list[5]])

#         if thumb_index_dist < 50 and util.get_angle(landmarks_list[5], landmarks_list[6], landmarks_list[8]) > 90 :
#             move_mouse(index_finger_tip)

#         # LEFT CLICK
#         elif is_left_click(landmarks_list, thumb_index_dist):
#             mouse.press(Button.left)
#             mouse.release(Button.left)
#             cv2.putText(frame, "Left Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # RIGHT CLICK
#         elif is_right_click(landmarks_list, thumb_index_dist):
#             mouse.press(Button.right)
#             mouse.release(Button.right)
#             cv2.putText(frame, "Right Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


#         # DOUBLE CLICK
#         elif is_double_click(landmarks_list, thumb_index_dist):
#             pyautogui.doubleClick()
#             cv2.putText(frame, "Double Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#         # SCREENSHOT
#         elif is_left_click(landmarks_list, thumb_index_dist):
#             im1 = pyautogui.screenshot()
#             label = random.randint(1,1000)
#             im1.save(f"my_screenshot_{label}.png")
#             cv2.putText(frame, "Right Click", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# def main():
#     cap = cv2.VideoCapture(0)
#     draw = mp.solutions.drawing_utils

#     try:
#         while cap.isOpened():
#             ret, frame = cap.read()

#             if not ret:
#                 break

#             frame = cv2.flip(frame, 1)
#             frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             processed = hands.process(frameRGB)

#             landmarks_list = []

#             if processed.multi_hand_landmarks:
#                 hand_landmarks = processed.multi_hand_landmarks[0]
#                 draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

#                 for lm in hand_landmarks.landmark:
#                     landmarks_list.append((lm.x, lm.y))

#             detect_gestures(frame, landmarks_list, processed)

#             cv2.imshow('Frame', frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         cap.release()
#         cv2.destroyAllWindows()



# if __name__ == '__main__':
#     main()

"""
Hand-Mouse Controller (final)
- Matches the 5 gestures exactly:
  1) index open, middle open, thumb bent -> MOVE (yellow)
  2) index open, middle open, thumb open -> STOP (white)
  3) index open, middle bent, thumb open -> RIGHT CLICK (red)
  4) index bent, middle open, thumb open -> LEFT CLICK (green)
  5) index bent, middle bent, thumb bent -> SCREENSHOT (blue)

Features:
- Smoothing for cursor movement
- Cooldowns for clicks and screenshot
- Optional drag-and-drop (pinch-to-drag while rule#1 active)
- Debug overlay with angles and thumb-index distance (toggle with 'd')
- Quit with 'q'
"""

# import cv2
# import mediapipe as mp
# import pyautogui
# from pynput.mouse import Button, Controller
# import random

# # Initialize mouse and screen size
# mouse = Controller()
# screen_w, screen_h = pyautogui.size()

# # Mediapipe setup
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7,
#                        min_tracking_confidence=0.7,
#                        max_num_hands=1)
# draw = mp.solutions.drawing_utils


# def finger_states(lmList):
#     """
#     Return [thumb, index, middle] -> 1=open, 0=bent
#     """
#     thumb = 1 if lmList[4][0] > lmList[3][0] else 0
#     index = 1 if lmList[8][1] < lmList[6][1] else 0
#     middle = 1 if lmList[12][1] < lmList[10][1] else 0
#     return thumb, index, middle


# def move_mouse(x, y):
#     """Move mouse pointer to given (x,y) in normalized coords"""
#     screen_x = int(x * screen_w)
#     screen_y = int(y * screen_h)
#     pyautogui.moveTo(screen_x, screen_y)


# # Start webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)  # Mirror image
#     h, w, _ = frame.shape

#     # Convert to RGB for mediapipe
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             # Collect landmarks
#             lmList = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

#             # Detect finger states
#             thumb, index, middle = finger_states(lmList)

#             # Rule 1: Move Arrow
#             if index == 1 and middle == 1 and thumb == 0:
#                 move_mouse(lmList[8][0] / w, lmList[8][1] / h)
#                 cv2.putText(frame, "Moving", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#             # Rule 2: Stop Arrow
#             elif index == 1 and middle == 1 and thumb == 1:
#                 cv2.putText(frame, "Stopped", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#             # Rule 3: Right Click
#             elif index == 1 and middle == 0 and thumb == 1:
#                 mouse.click(Button.right, 1)
#                 cv2.putText(frame, "Right Click", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#             # Rule 4: Left Click
#             elif index == 0 and middle == 1 and thumb == 1:
#                 mouse.click(Button.left, 1)
#                 cv2.putText(frame, "Left Click", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

#             # Rule 5: Screenshot
#             elif index == 0 and middle == 0 and thumb == 0:
#                 filename = f"screenshot_{random.randint(100,999)}.png"
#                 pyautogui.screenshot(filename)
#                 cv2.putText(frame, f"Screenshot: {filename}", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#             # Rule 6: Double Click
#             elif index == 1 and middle == 0 and thumb == 0:
#                 mouse.click(Button.left, 2)
#                 cv2.putText(frame, "Double Click", (50, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)

#             # Draw hand landmarks
#             draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     # Show webcam feed
#     cv2.imshow("Hand Mouse Control", frame)

#     # Exit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import pyautogui
from pynput.mouse import Button, Controller
import random
import time

# Initialize mouse and screen size
mouse = Controller()
screen_w, screen_h = pyautogui.size()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7,
                       max_num_hands=1)
draw = mp.solutions.drawing_utils

# Cooldown timers
last_screenshot_time = 0
last_doubleclick_time = 0


def finger_states(lmList):
    """
    Return [thumb, index, middle] -> 1=open, 0=bent
    """
    thumb = 1 if lmList[4][0] > lmList[3][0] else 0
    index = 1 if lmList[8][1] < lmList[6][1] else 0
    middle = 1 if lmList[12][1] < lmList[10][1] else 0
    return thumb, index, middle


def move_mouse(x, y):
    """Move mouse pointer to given (x,y) in normalized coords"""
    screen_x = int(x * screen_w)
    screen_y = int(y * screen_h)
    pyautogui.moveTo(screen_x, screen_y)


# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape

    # Convert to RGB for mediapipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Collect landmarks
            lmList = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]

            # Detect finger states
            thumb, index, middle = finger_states(lmList)

            # Rule 1: Move Arrow
            if index == 1 and middle == 1 and thumb == 0:
                move_mouse(lmList[8][0] / w, lmList[8][1] / h)
                cv2.putText(frame, "Moving", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Rule 2: Stop Arrow
            elif index == 1 and middle == 1 and thumb == 1:
                cv2.putText(frame, "Stopped", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Rule 3: Right Click
            elif index == 1 and middle == 0 and thumb == 1:
                mouse.click(Button.right, 1)
                cv2.putText(frame, "Right Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Rule 4: Left Click
            elif index == 0 and middle == 1 and thumb == 1:
                mouse.click(Button.left, 1)
                cv2.putText(frame, "Left Click", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Rule 5: Screenshot (with 2s cooldown)
            elif index == 0 and middle == 0 and thumb == 0:
                current_time = time.time()
                if current_time - last_screenshot_time > 2:
                    filename = f"screenshot_{random.randint(100,999)}.png"
                    pyautogui.screenshot(filename)
                    cv2.putText(frame, f"Screenshot: {filename}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    last_screenshot_time = current_time

            # Rule 6: Double Click (with 2s cooldown)
            elif index == 0 and middle == 0 and thumb == 1:
                current_time = time.time()
                if current_time - last_doubleclick_time > 2:
                    mouse.click(Button.left, 2)
                    cv2.putText(frame, "Double Click", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                    last_doubleclick_time = current_time


            # Draw hand landmarks
            draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Show webcam feed
    cv2.imshow("Hand Mouse Control", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
