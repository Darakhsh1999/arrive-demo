import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import re
import os

def run_knn_classifier():
    """
    Implements and evaluates a KNN classifier on the aggregated user data.
    """
    # --- 1. Load Data ---
    # Construct the path to the CSV file relative to the script's location
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, 'aggregated-user-data.csv')
    
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        # As a fallback, try to load from the current working directory
        try:
            data = pd.read_csv('aggregated-user-data.csv')
        except FileNotFoundError:
            print("Error: Could not find 'aggregated-user-data.csv' in src/ or current directory.")
            return

    # --- 2. Preprocessing ---
    # Drop the user ID as it's not a feature
    if 'parkinguser_id' in data.columns:
        data = data.drop('parkinguser_id', axis=1)

    # Convert target variable 'account_type' to integer
    if 'account_type' in data.columns:
        data['account_type'] = data['account_type'].astype(int)
        y = data['account_type']
        X = data.drop('account_type', axis=1)
    else:
        print("Error: 'account_type' column not found.")
        return

    # --- 3. Feature Engineering for 'area_type' ---
    if 'area_type' in X.columns:
        # This function parses the string to a list of floats.
        def parse_array_from_string(s):
            numbers = re.findall(r'-?[-+]?\d*\.?\d+', str(s))
            return [float(n) for n in numbers]

        try:
            TARGET_DIM = 7
            area_type_features = X['area_type'].apply(parse_array_from_string)

            # Pad or truncate each list to the target dimension
            processed_features = []
            for lst in area_type_features:
                if len(lst) > TARGET_DIM:
                    processed_features.append(lst[:TARGET_DIM])  # Truncate
                else:
                    # Pad with zeros if shorter
                    processed_features.append(lst + [0] * (TARGET_DIM - len(lst)))

            # Create a DataFrame from the processed features
            area_type_df = pd.DataFrame(
                processed_features, 
                index=X.index, 
                columns=[f'area_type_{i}' for i in range(TARGET_DIM)]
            )
            
            # Drop the original 'area_type' column and concatenate the new features
            X = X.drop('area_type', axis=1)
            X = pd.concat([X, area_type_df], axis=1)

        except Exception as e:
            print(f"Error processing 'area_type' column: {e}")
            return
    
    # --- 4. Model Pipeline ---
    # Create a pipeline that standardizes the features and then applies the KNN classifier.
    # StandardScaler standardizes features by removing the mean and scaling to unit variance.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    # --- 5. Evaluation ---
    # Leave-One-Out Cross-Validation (LOOCV)
    # In LOOCV, each sample is used once as a test set while the remaining samples form the training set.
    # This is repeated for each sample in the dataset.
    loo = LeaveOneOut()
    
    # 'cross_val_score' evaluates the model using the LOOCV strategy.
    # 'n_jobs=-1' uses all available CPU cores to speed up the computation.
    scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy', n_jobs=-1)

    # --- 6. Results ---
    print("--- KNN Classifier Evaluation ---")
    print(f"Data samples: {len(X)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"K (n_neighbors): 5")
    print("\nEvaluating with Leave-One-Out Cross-Validation...")
    print(f"\nMean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation of Accuracy: {scores.std():.4f}")

if __name__ == '__main__':
    run_knn_classifier()
