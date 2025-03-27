import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

def preprocess_data(file_path, augment_data=False):    
    df = pd.read_csv(file_path)    
    
    df = df.dropna()
    print("Missing values removed")    
    
    scaler = MinMaxScaler()
    feature_cols = df.columns[1:]
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    print("Data normalized")    
   
    X = df.iloc[:, 1:].values  
    y = df.iloc[:, 0].values   
    oversampler = RandomOverSampler()
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
    print("Dataset balanced")
       
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    print("Data split into training and testing sets")    
    
    if augment_data:
        noise = np.random.normal(0, 0.01, X_train.shape) 
        X_train = np.vstack((X_train, X_train + noise))
        y_train = np.concatenate((y_train, y_train))
        print(" Data augmentation applied")
    
   
    joblib.dump(scaler, "scaler.pkl")
    np.save("X_train.npy", X_train)
    np.save("X_test.npy", X_test)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)    
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    file_path = "hand_landmarks.csv" 
    preprocess_data(file_path, augment_data=True)
