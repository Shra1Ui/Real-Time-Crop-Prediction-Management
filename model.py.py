
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('/content/Crop_recommendation.csv')
data.head()

data.shape

data.isnull().sum()

"""# EDA"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Crop_recommendation.csv')

# Optional: for better plot visuals
sns.set(style="whitegrid")

plt.figure(figsize=(7, 4))
sns.histplot(df['rainfall'], kde=True, color='skyblue', bins=30)
plt.title('Distribution of Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10,4))
sns.boxplot(x=df['ph'], color='lightgreen')
plt.title('Boxplot of pH Levels')
plt.xlabel('pH')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
sns.countplot(y='label', data=df, order=df['label'].value_counts().index, palette='Set2')
plt.title('Count of Each Crop')
plt.xlabel('Count')
plt.ylabel('Crop Label')
plt.show()

#split features and lab
x = data.iloc[:,:-1] # features
y= data.iloc[:,-1] #labels

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

accuracy = model.score(X_test,y_test)

print(accuracy)

new_features = [[36,58,25,28.66,59.32891,8.3991,36.9263]]
predicted_crop= model.predict(new_features)
print("predict crop :", predicted_crop)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Predict
log_preds = log_model.predict(X_test)

# Accuracy
log_acc = accuracy_score(y_test, log_preds)
print("Logistic Regression Accuracy:", log_acc)

# Single prediction
new_features = [[36,58,25,28.66,59.32891,8.3991,36.9263]]
log_crop = log_model.predict(new_features)
print("Predicted crop (Logistic Regression):", log_crop)

from sklearn.neighbors import KNeighborsClassifier

# Model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Predict
knn_preds = knn_model.predict(X_test)

# Accuracy
knn_acc = accuracy_score(y_test, knn_preds)
print("KNN Accuracy:", knn_acc)

# Single prediction
knn_crop = knn_model.predict(new_features)
print("Predicted crop (KNN):", knn_crop)

from sklearn.svm import SVC

# Model
svm_model = SVC()
svm_model.fit(X_train, y_train)

# Predict
svm_preds = svm_model.predict(X_test)

# Accuracy
svm_acc = accuracy_score(y_test, svm_preds)
print("SVM Accuracy:", svm_acc)

# Single prediction
svm_crop = svm_model.predict(new_features)
print("Predicted crop (SVM):", svm_crop)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Store models and names
models = {
    "Random Forest": model,
    "Logistic Regression": log_model,
    "KNN": knn_model,
    "SVM": svm_model
}

# Loop through and evaluate
for name, clf in models.items():
    print(f"\n------ {name} ------")
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("Classification Report:")
    print(classification_report(y_test, preds))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Example for Random Forest
rf_preds = model.predict(X_test)
cm = confusion_matrix(y_test, rf_preds)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=False, cmap='Blues')  # Set annot=True if you want numbers
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestClassifier()

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# GridSearchCV with 3-fold cross-validation
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
grid.fit(X_train, y_train)

# Best model
best_rf = grid.best_estimator_

# Evaluate
accuracy = best_rf.score(X_test, y_test)
print("Tuned Random Forest Accuracy:", accuracy)
print("Best Parameters:", grid.best_params_)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_knn = GridSearchCV(knn, param_grid_knn, cv=3, scoring='accuracy')
grid_knn.fit(X_train, y_train)

best_knn = grid_knn.best_estimator_
print("Best KNN Accuracy:", best_knn.score(X_test, y_test))
print("Best KNN Params:", grid_knn.best_params_)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

svm = SVC()

param_grid_svm = {
    'C': [0.1, 1],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale']
}

# Using parallel processing with n_jobs=-1
grid_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring='accuracy', n_jobs=-1)
grid_svm.fit(X_train, y_train)

best_svm = grid_svm.best_estimator_
print("Best SVM Accuracy:", best_svm.score(X_test, y_test))
print("Best SVM Params:", grid_svm.best_params_)

import pickle

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

import pickle

# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)