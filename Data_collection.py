import cv2 as cv
import mediapipe as mp
import string
import os
import numpy as np

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mp_drawing = mp.solutions.drawing_utils

letters = list(string.ascii_uppercase)
Path = "DATA"

for letter in letters:
    os.makedirs(os.path.join(Path, letter), exist_ok=True)

sequence = 15
samples = 30


capture = cv.VideoCapture(0)

for letter in letters:
    print(f"Collecting data for {letter}:")

    for seq in range(samples):
        frames = []

        for i in range(sequence):
            ret, frame = capture.read()
            RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            result = hands.process(RGB_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)
                    hand_data = [coord for point in hand_landmarks.landmark for coord in (point.x, point.y, point.z)]
                    frames.append(hand_data)
            cv.imshow("DATA", cv.flip(frame, 1))
            if cv.waitKey(1) == ord('q'):
                break

        np.save(os.path.join(Path, letter, f"{seq}.npy"), frames)

capture.release()
cv.destroyAllWindows()

