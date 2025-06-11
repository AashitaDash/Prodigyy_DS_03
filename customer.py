import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set a random seed for reproducibility
np.random.seed(42)

# --- 1. Simulate a Bank Marketing-like Dataset ---
# This synthetic dataset mimics the structure and types of features
# found in the actual Bank Marketing dataset to make the code runnable
# without requiring an external download.

num_samples = 2000 # Number of simulated customers

# Demographic and Behavioral Data
data = {
    'age': np.random.randint(18, 90, num_samples),
    'job': np.random.choice(['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
                             'retired', 'self-employed', 'services', 'student', 'technician',
                             'unemployed', 'unknown'], num_samples,
                            # Corrected probabilities for 'job' to sum to exactly 1.0
                            p=[0.12, 0.22, 0.04, 0.03, 0.07,
                               0.05, 0.03, 0.09, 0.02, 0.16,
                               0.03, 0.14]), # Adjusted the last value to ensure sum is 1.0
    'marital': np.random.choice(['married', 'single', 'divorced', 'unknown'], num_samples, p=[0.6, 0.28, 0.11, 0.01]),
    'education': np.random.choice(['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate',
                                   'professional.course', 'university.degree', 'unknown'], num_samples),
    'default': np.random.choice(['no', 'yes', 'unknown'], num_samples, p=[0.8, 0.05, 0.15]),
    'housing': np.random.choice(['no', 'yes', 'unknown'], num_samples),
    'loan': np.random.choice(['no', 'yes', 'unknown'], num_samples),
    'contact': np.random.choice(['cellular', 'telephone'], num_samples),
    'month': np.random.choice(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], num_samples),
    'day_of_week': np.random.choice(['mon', 'tue', 'wed', 'thu', 'fri'], num_samples),
    'duration': np.random.randint(0, 2000, num_samples), # Contact duration in seconds
    'campaign': np.random.randint(1, 60, num_samples), # Number of contacts during this campaign
    # Simplified pdays simulation to avoid complex indexing for small num_samples
    'pdays': np.random.choice([-1, 999] + list(np.random.randint(0, 30, max(1, num_samples // 10))), num_samples),
    'previous': np.random.randint(0, 7, num_samples), # Number of contacts before this campaign
    'poutcome': np.random.choice(['failure', 'nonexistent', 'success'], num_samples, p=[0.1, 0.85, 0.05]),

    # Economic Indicators (simplified simulation)
    'emp.var.rate': np.random.normal(0, 1, num_samples).round(1),
    'cons.price.idx': np.random.normal(93, 1, num_samples).round(2),
    'cons.conf.idx': np.random.normal(-40, 5, num_samples).round(2),
    'euribor3m': np.random.normal(2, 2, num_samples).round(3),
    'nr.employed': np.random.normal(5000, 100, num_samples).round(0),

    # Target variable 'y' (customer subscribed to term deposit: 'yes' or 'no')
    # Introduce some imbalance, typically 'no' is more frequent
    'y': np.random.choice(['no', 'yes'], num_samples, p=[0.88, 0.12])
}

df = pd.DataFrame(data)

print("--- Simulated Dataset Created ---")
print(df.head())
print("\nDataset Info:")
df.info()
print("\n")

# --- 2. Preprocessing ---

# Separate target variable 'y' from features
X = df.drop('y', axis=1)
y = df['y']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Numerical Features: {numerical_features}")
print(f"Categorical Features: {categorical_features}\n")

# Create preprocessing pipelines for numerical and categorical features
# For numerical features, we'll just pass them through as Decision Trees don't require scaling.
# For categorical features, we'll use OneHotEncoder to convert them into numerical format.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features), # 'passthrough' means no transformation
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# Encode the target variable 'y' ('no' -> 0, 'yes' -> 1)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Encoded target variable mapping: {list(label_encoder.classes_)} -> {label_encoder.transform(label_encoder.classes_)}")
print("\nTarget variable encoded.")

# --- 3. Train-Test Split ---
# Split the dataset into training and testing sets to evaluate model performance
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
print("\n")

# --- 4. Model Training ---

# Create a pipeline that first preprocesses the data and then trains the Decision Tree Classifier
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5)) # max_depth to avoid overfitting
                                ])

print("Training Decision Tree Classifier...")
model_pipeline.fit(X_train, y_train)
print("Training complete.")
print("\n")

# --- 5. Evaluation ---

print("--- Model Evaluation ---")
y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary') # For binary classification
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10, rotation=0)
plt.tight_layout()
plt.show()


# --- 6. Visualize the Decision Tree (Optional) ---
# This part requires graphviz to render well for larger trees,
# but plot_tree from sklearn can render a basic version directly.

print("--- Visualizing Decision Tree (Simplified) ---")
# Get the trained decision tree model from the pipeline
decision_tree_model = model_pipeline.named_steps['classifier']
# Get feature names after one-hot encoding
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names)

plt.figure(figsize=(20, 10)) # Adjust size for better readability
plot_tree(decision_tree_model,
          feature_names=all_feature_names,
          class_names=label_encoder.classes_,
          filled=True,
          rounded=True,
          fontsize=8)
plt.title('Decision Tree Classifier Visualization (Max Depth 5)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("\n--- Decision Tree Classifier Task Complete ---")
print("Model trained, evaluated, and visualization attempted.")
