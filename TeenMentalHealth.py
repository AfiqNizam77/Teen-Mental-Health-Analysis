import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
df = pd.read_csv('Teen_Mental_Health_Dataset.csv')

# Encode categorical variables (Gender, Platform, Social Level)
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['platform_usage'] = le.fit_transform(df['platform_usage'])
df['social_interaction_level'] = le.fit_transform(df['social_interaction_level'])

# Define Features (X) and Target (y)
X = df.drop('depression_label', axis=1)
y = df['depression_label']

# Split data into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Artificial Neural Network
ann_model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)


ann_model.fit(X_train, y_train)

# 3. Report and Visualization
y_pred = ann_model.predict(X_test)

print(f"Final Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 4. Visualization
# Diagram A: Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Depression', 'Depression'],
            yticklabels=['No Depression', 'Depression'])
plt.title('ANN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Diagram B: Learning Curve
plt.figure(figsize=(10,6))
plt.plot(ann_model.loss_curve_)
plt.title('Learning Curve')
plt.xlabel('Epoch')
plt.ylabel('Error Value')
plt.grid(True)
plt.show()