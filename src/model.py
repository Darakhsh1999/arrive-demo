import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(csv_path):
    """
    Loads the CSV data and performs basic preprocessing and feature engineering.
    Returns X (features) and y (target).
    """
    df = pd.read_csv(csv_path)

    # Feature engineering: parking duration (in minutes)
    df['parking_start_time'] = pd.to_datetime(df['parking_start_time'])
    df['parking_end_time'] = pd.to_datetime(df['parking_end_time'])
    df['duration_minutes'] = (df['parking_end_time'] - df['parking_start_time']).dt.total_seconds() / 60
    df['start_hour'] = df['parking_start_time'].dt.hour
    df['day_of_week'] = df['parking_start_time'].dt.dayofweek

    # Target encoding: 'account_type' (private=0, corporate=1)
    df['account_type'] = df['account_type'].map({'private': 0, 'corporate': 1})

    # Select features for modeling
    features = [
        'area_type', 'parking_fee', 'currency', 'lat', 'lon',
        'duration_minutes', 'start_hour', 'day_of_week'
    ]
    X = df[features]
    y = df['account_type']
    return X, y

def train_and_evaluate(X, y):
    """
    Splits the data, fits a Logistic Regression model, and prints evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing for numeric and categorical features
    numeric_features = ['parking_fee', 'lat', 'lon', 'duration_minutes', 'start_hour', 'day_of_week']
    categorical_features = ['area_type', 'currency']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Create pipeline with preprocessing and Logistic Regression
    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    return clf

if __name__ == "__main__":
    # Example usage
    # Replace 'assignment_sample_data.csv' with the actual path to your data
    X, y = load_and_preprocess_data("assignment-sample-data.csv")
    train_and_evaluate(X, y)
