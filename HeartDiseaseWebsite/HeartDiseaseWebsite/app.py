
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_uci.csv")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(STATIC_DIR, exist_ok=True)

NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg",
    "exang", "slope", "ca", "thal"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Normalize common categorical values from different CSV versions.
CATEGORY_MAPS = {
    "sex": {
        "0": "female", "f": "female", "female": "female",
        "1": "male", "m": "male", "male": "male"
    },
    "cp": {
        "1": "typical angina",
        "2": "atypical angina",
        "3": "non-anginal",
        "4": "asymptomatic",
        "typical angina": "typical angina",
        "atypical angina": "atypical angina",
        "non-anginal": "non-anginal",
        "non-anginal pain": "non-anginal",
        "asymptomatic": "asymptomatic"
    },
    "fbs": {
        "0": "false", "false": "false", "no": "false",
        "1": "true", "true": "true", "yes": "true"
    },
    "restecg": {
        "0": "normal",
        "1": "st-t abnormality",
        "2": "left ventricular hypertrophy",
        "normal": "normal",
        "st-t abnormality": "st-t abnormality",
        "st-t wave abnormality": "st-t abnormality",
        "left ventricular hypertrophy": "left ventricular hypertrophy"
    },
    "exang": {
        "0": "no", "no": "no", "false": "no",
        "1": "yes", "yes": "yes", "true": "yes"
    },
    "slope": {
        "1": "upsloping",
        "2": "flat",
        "3": "downsloping",
        "upsloping": "upsloping",
        "flat": "flat",
        "downsloping": "downsloping"
    },
    "thal": {
        "3": "normal",
        "6": "fixed defect",
        "7": "reversible defect",
        "normal": "normal",
        "fixed defect": "fixed defect",
        "reversible defect": "reversible defect"
    },
    "ca": {}
}


def normalize_category(column, value):
    """Normalize category values for consistent training and prediction."""
    if pd.isna(value):
        return np.nan

    value = str(value).strip().lower()

    # Normalize values such as 0.0 to 0
    try:
        number = float(value)
        if number.is_integer():
            value = str(int(number))
    except ValueError:
        pass

    return CATEGORY_MAPS.get(column, {}).get(column, {}).get(
        value, value
    ) if False else CATEGORY_MAPS.get(column, {}).get(value, value)


# Load dataset
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

# Support the common UCI column name "thalch"
if "thalch" in df.columns and "thalach" not in df.columns:
    df.rename(columns={"thalch": "thalach"}, inplace=True)

# Identify target column
if "num" in df.columns:
    target_column = "num"
elif "target" in df.columns:
    target_column = "target"
else:
    raise ValueError(
        "The dataset must contain a 'num' or 'target' column."
    )

# Convert target to binary:
# 0 = no disease; values above 0 = disease present
target_values = pd.to_numeric(df[target_column], errors="coerce")
valid_target = target_values.notna()

df = df.loc[valid_target].copy()
y = (target_values.loc[valid_target] > 0).astype(int)

# Prepare features
missing_features = [col for col in FEATURES if col not in df.columns]
if missing_features:
    raise ValueError(
        f"Missing required dataset columns: {missing_features}"
    )

X = df[FEATURES].copy()

for column in NUMERIC_FEATURES:
    X[column] = pd.to_numeric(X[column], errors="coerce")

for column in CATEGORICAL_FEATURES:
    X[column] = X[column].apply(
        lambda value: normalize_category(column, value)
    )

# Remove rows where all feature values are missing
valid_features = X.notna().any(axis=1)
X = X.loc[valid_features].copy()
y = y.loc[valid_features].copy()
df = df.loc[valid_features].copy()

if len(X) < 2:
    raise ValueError("The dataset does not contain enough usable records.")

# Preprocess numeric and categorical columns separately
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
])

# Train the KNN classifier
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(
        n_neighbors=min(5, len(X)),
        weights="distance"
    ))
])

model.fit(X, y)


def create_graphs():
    """Create charts using the actual dataset records."""

    # Graph 1: Actual binary target distribution
    counts = y.value_counts().reindex([0, 1], fill_value=0)

    plt.figure(figsize=(7, 4))
    bars = plt.bar(
        ["No Disease", "Disease Present"],
        counts.values
    )

    plt.title("Heart Disease Distribution in Dataset")
    plt.xlabel("Target Class")
    plt.ylabel("Number of Patients")

    for bar, count in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(STATIC_DIR, "target_distribution.png"),
        dpi=150
    )
    plt.close()

    # Graph 2: Actual patient age distribution
    ages = pd.to_numeric(df["age"], errors="coerce").dropna()

    plt.figure(figsize=(7, 4))
    plt.hist(
        ages,
        bins=15,
        edgecolor="black"
    )

    plt.title("Age Distribution of Patients in Dataset")
    plt.xlabel("Age in Years")
    plt.ylabel("Number of Patients")
    plt.tight_layout()

    plt.savefig(
        os.path.join(STATIC_DIR, "age_distribution.png"),
        dpi=150
    )
    plt.close()


# Generate image files at startup.
# The HTML controls when they are displayed.
create_graphs()


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk_percentage = None
    error = None
    form_data = {}
    show_graphs = False

    if request.method == "POST":
        # Preserve every submitted field, including dropdown selections.
        for feature in FEATURES:
            form_data[feature] = request.form.get(
                feature, ""
            ).strip()

        try:
            for feature in FEATURES:
                if not form_data[feature]:
                    raise ValueError(
                        f"Please select or enter {feature}."
                    )

            input_row = {}

            for feature in NUMERIC_FEATURES:
                value = float(form_data[feature])

                if not np.isfinite(value):
                    raise ValueError(
                        f"Enter a valid number for {feature}."
                    )

                input_row[feature] = value

            for feature in CATEGORICAL_FEATURES:
                input_row[feature] = normalize_category(
                    feature, form_data[feature]
                )

            if not 1 <= input_row["age"] <= 120:
                raise ValueError("Age must be between 1 and 120.")

            if input_row["trestbps"] <= 0:
                raise ValueError(
                    "Resting blood pressure must be positive."
                )

            if input_row["chol"] < 0:
                raise ValueError(
                    "Cholesterol cannot be negative."
                )

            input_df = pd.DataFrame([input_row], columns=FEATURES)

            # Make prediction with the same preprocessing pipeline
            prediction = int(model.predict(input_df)[0])

            probabilities = model.predict_proba(input_df)[0]
            disease_index = list(model.classes_).index(1)

            risk_percentage = round(
                float(probabilities[disease_index]) * 100, 2
            )

            # Display graphs after a successful prediction
            show_graphs = True

        except (ValueError, TypeError) as exc:
            error = str(exc)

    return render_template(
        "index.html",
        prediction=prediction,
        risk_percentage=risk_percentage,
        error=error,
        form_data=form_data,
        show_graphs=show_graphs
    )


if __name__ == "__main__":
    app.run(debug=True)
