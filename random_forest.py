import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib


# Load features and labels
df = pd.read_csv('phishing_features.csv')
X = df.drop('Label', axis=1)
y = df['Label']  # Now expects 'good' for safe and 'bad' for phishing

# If you want to map 'good'/'bad' to 0/1, uncomment below:
# y = y.map({'good': 0, 'bad': 1})

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# RandomizedSearchCV for Random Forest hyperparameter tuning
param_dist = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,           # Number of parameter settings sampled
    cv=5,                # Increased folds from 3 to 5
    n_jobs=-1,
    verbose=2,
    scoring='accuracy',
    random_state=42
)
random_search.fit(X_train, y_train)
best_rf = random_search.best_estimator_

print("Best hyperparameters:", random_search.best_params_)

# Save the trained model to a file
joblib.dump(best_rf, 'random_forest_model.joblib')
print('Model saved as random_forest_model.joblib')


# Evaluate on test set
y_pred_rf = best_rf.predict(X_test)
print('Random Forest Test Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# Optionally, evaluate on validation set as well
print('Validation set results:')
print('Random Forest Validation Accuracy:', accuracy_score(y_val, best_rf.predict(X_val)))

# Print class labels for reference
print('Model classes:', best_rf.classes_)
