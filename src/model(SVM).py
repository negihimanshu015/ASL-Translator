import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from joblib import dump
from tqdm import tqdm

X_train = np.load("Data\\Processed\\X_train.npy", allow_pickle=True)
y_train = np.load("Data\\Processed\\y_train.npy", allow_pickle=True)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', probability=True)

print("Training SVM model...")
for i in tqdm(range(1, 101), desc="Training Progress", ncols=100):
    svm.fit(X_train[: i * len(X_train) // 100], y_train[: i * len(y_train) // 100])

accuracy = svm.score(X_val, y_val)
print(f"\n Model trained successfully! Accuracy: {accuracy:.2f}")

dump(svm, "models/sign_language_svm.pkl")

