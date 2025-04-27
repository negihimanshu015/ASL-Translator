import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.7)

dataset_path = os.path.join("dataset", "asl_alphabet_train")
output_csv = "Data/raw/hand_landmarks.csv"

if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        headers = ["Label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)]
        writer.writerow(headers)

for letter in sorted(os.listdir(dataset_path)):
    letter_path = os.path.join(dataset_path, letter)
    
    if os.path.isdir(letter_path): 
        print(f"Processing letter: {letter}")
        
        image_files = sorted(os.listdir(letter_path))

        for image_name in image_files:
            image_path = os.path.join(letter_path, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f" Error: Could not load {image_name}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:                    
                    landmarks = []
                    for point in hand_landmarks.landmark:
                        landmarks.extend([point.x, point.y, point.z])
                    
                    with open(output_csv, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([letter.upper()] + landmarks)

        print(f" {letter}: Processed {len(image_files)} images.")


