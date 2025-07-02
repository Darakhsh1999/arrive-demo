import pandas as pd
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

def run_knn_classifier(k: int = 5):
    """
    Implements and evaluates a KNN classifier on the aggregated user data.
    """

    ### Load CSV Data ###
    try:
        data = pd.read_csv('aggregated-user-data.csv') # (300, 15)
    except Exception as e:
        print(f"Error: The file 'aggregated-user-data.csv' was not found.")
        raise e
    
    
    ### Data Preprocessing ###

    # Drop the user ID as it's not a feature
    data = data.drop('parkinguser_id', axis=1) # (300, 14)

    # Convert target variable 'account_type' to integer and separate features and target (X,y)
    data['account_type'] = data['account_type'].astype(int)
    y = data['account_type'] # (300,)
    X = data.drop('account_type', axis=1) # (300, 13)


    ### Model Pipeline ###

    # Create a pipeline that standardizes the features and then applies the KNN classifier.
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()), # One could use StandardScaler() or Normalizer() instead
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])

    ### Model Evaluation ###

    # 'cross_val_score' evaluates the model using the LOOCV strategy.
    scores = cross_val_score(pipeline, X, y, cv=LeaveOneOut(), scoring='accuracy', n_jobs=-1)


    ### Sample prediction ###

    # Manual LOOCV to collect misclassified samples
    misclassified = []  # To store (index, true_label, predicted_label)
    misclassified_class0_indices = []  # To store indices of misclassified samples with class 0

    for idx, (train_index, test_index) in enumerate(LeaveOneOut().split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Fit scaler and model on training set only
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)

        if y_pred[0] != y_test.values[0]:
            misclassified.append({
                'index': test_index[0],
                'true_label': y_test.values[0],
                'predicted_label': y_pred[0]
            })
            if y_test.values[0] == 0:
                misclassified_class0_indices.append(test_index[0])

    # Analyze misclassified samples
    true_labels = [item['true_label'] for item in misclassified]
    class_counts = Counter(true_labels)
    print("\nMisclassified sample count by true class:")
    for label, count in class_counts.items():
        print(f"Class {label}: {count}")
    print(f"Total misclassified: {len(misclassified)}")

    # Print indices of misclassified samples with true label 0
    print(f"\nIndices of misclassified samples with true label 0: {[int(i) for i in misclassified_class0_indices]}")

    # Evaluation Results 
    print("--- KNN Classifier Evaluation ---")
    print(f"Data samples: {len(X)}")
    print(f"Number of features: {len(X.columns)}")
    print(f"K (n_neighbors): {k}")
    print("\nEvaluating with Leave-One-Out Cross-Validation...")
    print(f"\nMean Accuracy: {scores.mean():.4f}")
    print(f"Standard Deviation of Accuracy: {scores.std():.4f}")
    print(pipeline)


if __name__ == '__main__':
    k = 3
    run_knn_classifier(k)
