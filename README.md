# ASL-Translator

## Project Description
ASL-Translator is a machine learning project that translates American Sign Language (ASL) signs into text in real-time using a webcam. The system leverages MediaPipe for hand landmark detection, preprocesses the data, trains a Support Vector Machine (SVM) model, and performs live prediction of ASL signs.

## Project Structure
```
ASL-Translator/
├── Data/
│   ├── raw/                  
│   └── Processed/            
├── models/
│   ├── scaler.pkl            
│   └── sign_language_svm.pkl
├── src/
│   ├── cv.py                 
│   ├── landmark.py           
│   ├── model(SVM).py         
│   └── preprocess.py        
├── README.md                 
├── LICENSE                   
└── .gitignore                
```

## Installation and Setup

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ASL-Translator
   ```

2. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```
   Required packages include:
   - opencv-python
   - mediapipe
   - numpy
   - pandas
   - scikit-learn
   - imbalanced-learn
   - joblib
   - tqdm

3. Ensure you have a webcam connected for real-time prediction.

## Usage

**Note:** You need to provide and save the ASL alphabet image dataset yourself in the folder `dataset/asl_alphabet_train/` before running the landmark extraction script.

### 1. Extract Hand Landmarks from Images
Run the landmark extraction script to process the ASL alphabet image dataset and generate raw landmark data:
```
python src/landmark.py
```
This will create `Data/raw/hand_landmarks.csv`.

### 2. Preprocess Data
Preprocess the raw landmark data, normalize features, balance the dataset, and optionally augment data:
```
python src/preprocess.py
```
This generates processed training and testing datasets in `Data/Processed/` and saves the scaler to `models/scaler.pkl`.
python src/preprocess.py

### 3. Train the SVM Model
Train the Support Vector Machine model on the processed data:
```
python src/model(SVM).py
```
The trained model is saved as `models/sign_language_svm.pkl`.

### 4. Run Real-Time ASL Translator
Use your webcam to perform real-time ASL sign prediction:
```
python src/cv.py
```
Press `q` to quit the application.

## Data Description
- `Data/raw/`: Contains raw hand landmark coordinates extracted from ASL alphabet images.
- `Data/Processed/`: Contains normalized, balanced, and optionally augmented datasets split into training and testing sets.

## Model Information
- The project uses a linear kernel Support Vector Machine (SVM) classifier.
- The model is trained on hand landmark features extracted from images.
- Input features are normalized using MinMaxScaler.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for improvements or bug fixes.


