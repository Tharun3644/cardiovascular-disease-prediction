import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df = pd.read_csv("heart.csv")  # Replace with the correct file path

# Preview
print(df.head())
print(df.info())

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Data Preprocessing
# Optional: if any categorical columns are present, convert them using pd.get_dummies()
# Scaling
scaler = StandardScaler()
X = df.drop("target", axis=1)
y = df["target"]
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Visualization
sns.countplot(x='target', data=df)
plt.title("Target Variable Distribution")
plt.show()

sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Model Training and Evaluation
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.2f}")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds))
    print("="*50)

# Plotting Accuracy Comparison
plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.show()

# Final Model (choose the best one based on accuracy)
best_model_name = max(accuracies, key=accuracies.get)
print(f"Best model: {best_model_name} with accuracy {accuracies[best_model_name]:.2f}")
