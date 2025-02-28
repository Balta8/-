# Breast Cancer Classification using Machine Learning

## Overview
This project implements multiple machine learning models to classify breast cancer tumors as **Benign** or **Malignant** based on a publicly available dataset. The models used include:

- Naïve Bayes
- Decision Tree
- K-Nearest Neighbors (K-NN)
- Support Vector Machine (SVM)
- Multi-Layer Perceptron (MLP)
- Logistic Regression (Baseline)
- **Ensemble Learning Approaches:**
  - **Voting Classifier** (Combining multiple models using majority voting)
  - **Stacking Classifier** (Combining models with a meta-learner)

## Dataset
The dataset used is the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset from the UCI Machine Learning Repository. It consists of 30 numerical features extracted from digitized images of fine needle aspirates of breast masses.

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/)
- **Classes:** Benign (0), Malignant (1)
- **Features:** 30 computed properties (e.g., radius, texture, smoothness, etc.)

## Project Workflow

1. **Data Preprocessing:**
   - Loading and exploring the dataset
   - Handling missing values (if any)
   - Label encoding for categorical variables
   - Standardizing numerical features
   
2. **Exploratory Data Analysis (EDA):**
   - Diagnosis distribution visualization
   - Feature correlation heatmap
   
3. **Train-Test-Validation Split:**
   - Stratified splitting to maintain class balance across training, validation, and test sets.
   
4. **Model Training & Evaluation:**
   - Training individual models
   - Evaluating performance using Accuracy, Precision, Recall, and F1-score
   
5. **Dimensionality Reduction using PCA:**
   - Applying Principal Component Analysis (PCA)
   - Training and comparing models with and without PCA
   
6. **Ensemble Learning Approaches:**
   - Voting Classifier (Hard/Soft Voting)
   - Stacking Classifier with Logistic Regression as meta-model
   
7. **Explainability using SHAP:**
   - Understanding feature importance using SHAP values
   - Visualizing model contribution to predictions

## Results & Comparison
Each model's performance is compared using evaluation metrics. The **ensemble models** (Voting and Stacking) are expected to outperform individual models by leveraging multiple learners.

### SHAP Analysis
SHAP (SHapley Additive exPlanations) is used to interpret how each base model in the stacking classifier contributes to the final prediction. The analysis includes:
- Feature importance visualization
- Model contribution comparison before and after ensemble learning

## Installation & Usage

### Requirements
Make sure you have Python installed along with the required libraries. Install dependencies using:
```bash
pip install numpy pandas matplotlib seaborn shap scikit-learn
```

### Running the Project
Run the main script using:
```bash
python main.py
```
This will execute the full pipeline from data preprocessing to visualization and model comparison.

## Visualizations
- **Diagnosis Distribution:** Bar chart showing the number of benign vs. malignant cases.
- **Feature Correlation Heatmap:** Helps understand relationships between features.
- **Model Performance Comparison:** Line plot comparing accuracy, precision, recall, and F1-score.
- **SHAP Summary Plot:** Displays feature contributions to predictions.
- **Model Contribution Plot:** Shows how much each model in the stacking classifier contributes to predictions.

## Conclusion
The project demonstrates the effectiveness of machine learning in medical diagnosis. **Ensemble learning**, particularly stacking, improves prediction accuracy by combining multiple classifiers. **SHAP analysis** provides insight into model decision-making, improving interpretability.

## Future Improvements
- Implementing deep learning models (CNN, RNN) for better feature extraction.
- Exploring hyperparameter tuning for better model optimization.
- Integrating additional explainability techniques beyond SHAP.

---
### Author: Saleha
For any queries, feel free to reach out!
