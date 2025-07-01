import re
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def parse_array_from_string(s):
    numbers = re.findall(r'-?[-+]?\d*\.?\d+', str(s))
    return [float(n) for n in numbers]

def run_knn_classifier():
    """
    Implements and evaluates a KNN classifier on the aggregated user data.
    """

    ### Load Data ###
    try:
        data_path = 'aggregated-user-data.csv'
        data = pd.read_csv(data_path) # (300, 9)
    except Exception as e:
        print(f"Error: The file '{data_path}' was not found.")
        raise e
    
    
    ### Preprocessing ###

    # Drop the user ID as it's not a feature
    data = data.drop('parkinguser_id', axis=1) # (300, 8)

    # Convert target variable 'account_type' to integer and separate features and target (X,y)
    data['account_type'] = data['account_type'].astype(int)
    y = data['account_type'] # (300,)
    X = data.drop('account_type', axis=1) # (300, 7)

    ### Feature Engineering for area_type ###

    try:
        # Parse the string representation of lists into actual lists of floats
        area_type_features = X['area_type'].apply(parse_array_from_string).tolist()

        # --- Diagnostic Check ---
        # Find and print vectors with lengths other than 7 to investigate
        print("\n--- Checking Embedding Dimensions ---")
        found_mismatch = False
        for i, lst in enumerate(area_type_features):
            if len(lst) != 7:
                print(f"Row index {X.index[i]}: Found vector of length {len(lst)}.")
                found_mismatch = True
        
        if not found_mismatch:
            print("All embedding vectors have a length of 7.")
        print("-------------------------------------\n")
        # --- End Diagnostic Check ---

        # Determine the dimension from the longest vector to avoid data loss
        if any(area_type_features):
            max_dim = max(len(lst) for lst in area_type_features if lst)

            # Pad each list to the max dimension to ensure all vectors are of equal length
            processed_features = [lst + [0] * (max_dim - len(lst)) for lst in area_type_features]

            # Create a DataFrame from the processed features
            area_type_df = pd.DataFrame(
                processed_features,
                index=X.index,
                columns=[f'area_type_{i}' for i in range(max_dim)]
            )

            # Drop the original 'area_type' column and concatenate the new feature columns
            X = X.drop('area_type', axis=1)
            X = pd.concat([X, area_type_df], axis=1)
        else:
            # If the column is empty or contains no valid vectors, just drop it
            X = X.drop('area_type', axis=1)

    except Exception as e:
        print(f"Error processing 'area_type' column: {e}")
        return
    

    ### Model Pipeline ###

    # Create a pipeline that standardizes the features and then applies the KNN classifier.
    # StandardScaler standardizes features by removing the mean and scaling to unit variance.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])

    ### Evaluation ###

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
