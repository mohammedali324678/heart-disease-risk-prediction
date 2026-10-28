# Heart Disease Risk Prediction Website

A Flask website that uses a K-Nearest Neighbors (KNN) classifier implemented from scratch in Python. Pandas, NumPy, and scikit-learn are used for data handling and feature scaling; the KNN distance calculation and voting are custom Python code.

## Files

- `app.py` - Flask application, dataset preparation, and custom KNN implementation
- `templates/index.html` - patient form and prediction result
- `heart.csv` - place your Kaggle/UCI heart disease CSV here
- `requirements.txt` - Python dependencies

## Dataset requirements

Use a CSV with the standard UCI 13 input columns:

`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

The target column should be `target` or `num`. The target is converted to binary: 0 means no disease, values greater than 0 mean disease present.

Some Kaggle copies use different feature names or category coding. This project expects the original 13-feature UCI column names and codes. Check the dataset's data dictionary before using patient values.

## Run locally

1. Install Python 3.10 or newer.
2. Put `heart.csv` in this folder.
3. Open a terminal in this folder.
4. Install dependencies:

   `python -m pip install -r requirements.txt`

5. Start the website:

   `python app.py`

6. Open `http://127.0.0.1:5000` in your browser.

The app trains the model from the CSV when it starts. Internet access is not needed after dependencies and the dataset are available.

## Important

This is an educational demonstration, not a medical device or clinically validated diagnosis. Do not use its predictions to make healthcare decisions. The built-in Flask development server is for local development only.
