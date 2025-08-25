"""
Virtual Hand Controll Mouse 
- Matches the 5 gestures exactly:
  1) index open, middle open, thumb bent -> MOVE ARROW
  2) index open, middle open, thumb open -> STOP MOVING ARROW
  3) index open, middle bent, thumb open -> RIGHT CLICK 
  4) index bent, middle open, thumb open -> LEFT CLICK
  5) index bent, middle bent, thumb bent -> SCREENSHOT
  6) index bent, middle bent, thumb open -> DOUBLE CLICK
- Quit with 'q'
"""
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
