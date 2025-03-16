import cv2 as cv
import mediapipe as mp

capture = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = capture.read()
    RGB_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(RGB_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mpHands.HAND_CONNECTIONS)

    cv.imshow("frame",cv.flip(frame, 1))

    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows