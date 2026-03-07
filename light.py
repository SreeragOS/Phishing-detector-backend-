

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import joblib

# Load features and labels
df = pd.read_csv('phishing_features.csv')
X = df.drop('Label', axis=1)
y = df['Label']

# Split data: 80% train, 10% validation, 10% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)



# RandomizedSearchCV for LightGBM hyperparameter tuning
param_dist = {
	'n_estimators': [100, 200, 300],
	'max_depth': [5, 10, 20, None],
	'learning_rate': [0.01, 0.05, 0.1],
	'num_leaves': [15, 31, 63],
	'min_child_samples': [10, 20, 30]
}

lgbm = lgb.LGBMClassifier(random_state=42)
random_search = RandomizedSearchCV(
	estimator=lgbm,
	param_distributions=param_dist,
	n_iter=10,  # Number of parameter settings sampled
	cv=3,
	n_jobs=-1,
	verbose=2,
	scoring='accuracy',
	random_state=42
)
random_search.fit(X_train, y_train)
best_lgbm = random_search.best_estimator_
print("Best hyperparameters:", random_search.best_params_)

# Save the trained model to a file
joblib.dump(best_lgbm, 'lightgbm_model.joblib')
print('Model saved as lightgbm_model.joblib')

y_pred_lgbm = best_lgbm.predict(X_test)
print('LightGBM Test Accuracy:', accuracy_score(y_test, y_pred_lgbm))
print(classification_report(y_test, y_pred_lgbm))




# Evaluate on validation set
print('Validation set results:')
y_pred_val = best_lgbm.predict(X_val)
print('LightGBM Validation Accuracy:', accuracy_score(y_val, y_pred_val))
print(classification_report(y_val, y_pred_val))

# Print class labels for reference
print('Model classes:', best_lgbm.classes_)
