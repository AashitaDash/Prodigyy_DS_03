# Prodigyy_DS_03

Customer Purchase Prediction using Decision Tree Classifier
This repository contains Python code for building and evaluating a Decision Tree Classifier to predict whether a customer will purchase a product or service. The project is based on the principles of the Bank Marketing dataset.

Project Goal
The primary objective of this project is to develop a predictive model that can classify customers into two categories: those who are likely to purchase a product/service and those who are not, based on their demographic and behavioral data. This can help businesses target their marketing efforts more effectively.

Project Structure
.
├── customer_purchase_prediction.py  # Main Python script for the project
└── README.md

Getting Started
Prerequisites
To run this code, you'll need Python installed along with the following libraries:

pandas

numpy

scikit-learn

matplotlib

seaborn

You can install them using pip:

pip install pandas numpy scikit-learn matplotlib seaborn

Dataset
The code is designed to be runnable out-of-the-box by simulating a dataset that mimics the characteristics of the Bank Marketing dataset. This means you don't need to download any external files to run the provided script.

However, if you wish to use the actual dataset, you can download it from the UCI Machine Learning Repository:

Bank Marketing Dataset

If you download the dataset, you would typically use the bank-additional-full.csv file. You would then modify the customer_purchase_prediction.py script to load this file instead of generating a simulated one.

Running the Code
Save the provided Python code (from the customer_purchase_prediction.py section below) into a file named customer_purchase_prediction.py.

Open your terminal or command prompt.

Navigate to the directory where you saved the file.

Execute the script using Python:

python customer_purchase_prediction.py

The script will print various outputs to the console, including data information, preprocessing steps, model evaluation metrics, and a confusion matrix. It will also display a confusion matrix plot and a simplified visualization of the decision tree. Close each plot window to proceed.

Code Overview (customer_purchase_prediction.py)
The customer_purchase_prediction.py script performs the following key steps:

Dataset Simulation: Generates a synthetic dataset to ensure the code is immediately runnable for demonstration purposes. This dataset includes various demographic, behavioral, and economic features, along with a binary target variable (y) indicating a purchase.

Preprocessing:

Separates features (X) and the target variable (y).

Identifies numerical and categorical features.

Uses OneHotEncoder for categorical features and LabelEncoder for the target variable to convert them into a numerical format suitable for machine learning models.

Employs ColumnTransformer and Pipeline for a streamlined and reproducible preprocessing workflow.

Train-Test Split: Divides the dataset into training and testing sets to properly evaluate the model's generalization performance. Stratified splitting is used to maintain the proportion of target classes.

Model Training: Initializes and trains a DecisionTreeClassifier. A max_depth is set to control complexity and prevent overfitting.

Evaluation: Predicts on the test set and calculates various classification metrics:

Accuracy: Overall correctness of predictions.

Precision: Proportion of positive identifications that were actually correct.

Recall: Proportion of actual positives that were identified correctly.

F1-Score: Harmonic mean of precision and recall.

Confusion Matrix: A table showing the counts of true positive, true negative, false positive, and false negative predictions.

Visualization:

Displays a heatmap of the confusion matrix for better interpretability.

Generates a simplified plot of the trained Decision Tree, showing its structure and decision rules (for trees with max_depth=5).

Learnings and Insights
Data Preparation is Key: Effective preprocessing, especially for mixed-type datasets, is crucial for model performance. Techniques like one-hot encoding for categorical features are fundamental.

Decision Tree Interpretability: Decision Trees offer a clear, tree-like structure that makes it relatively easy to understand the rules the model uses to make predictions. This interpretability is highly valuable in business contexts.

Model Evaluation Beyond Accuracy: Relying solely on accuracy can be misleading, especially with imbalanced datasets. Metrics like precision, recall, and F1-score provide a more comprehensive view of model performance for both classes.

Pipeline for Reproducibility: Using sklearn.pipeline helps create a robust and shareable workflow, combining preprocessing and modeling steps into a single, cohesive object.

Contributing
Feel free to fork this repository, open issues, or submit pull requests. Any suggestions for improvements or alternative approaches are welcome!

License
This project is open source and available under the MIT License.
