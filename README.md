📊 HR Employee Attrition Prediction

📌 Project Overview

Employee attrition is a critical challenge for organizations, as losing skilled employees increases recruitment cost and reduces productivity.
This project builds a Machine Learning classification model to predict whether an employee is likely to leave the organization based on HR-related features.

The model is trained using Logistic Regression with proper preprocessing pipelines and deployed using Hugging Face Spaces.

🎯 Objectives

- Analyze employee data to understand attrition patterns
- Build a robust ML pipeline for prediction
- Evaluate model performance using multiple classification metrics
- Deploy the trained model for real-time prediction

🗂 Dataset

Source: HR Employee Attrition Dataset

Target Variable: Attrition

- Yes → 1
- No → 0

Feature Types

- Numerical Features: Age, MonthlyIncome, TotalWorkingYears, etc.
- Binary Features: Gender, OverTime
- Categorical Features: Department, JobRole, EducationField, MaritalStatus

🧠 Machine Learning Approach

🔹 Preprocessing

- Missing value handling using SimpleImputer
- Feature scaling using StandardScaler
- Categorical encoding using:
- LabelEncoder (binary features)
- OneHotEncoder (multi-class categorical features)
- Combined using ColumnTransformer

🔹 Model

Algorithm: Logistic Regression

Hyperparameter:

C = 93

Implemented using Scikit-learn Pipeline

📈 Model Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision (Weighted)
- Recall (Weighted)
- F1 Score (Weighted)
- ROC-AUC Score
- Confusion Matrix
- Classification Report

💾 Model Saving

The trained pipeline is saved using pickle:

pickle.dump(model, file)


and loaded during deployment for inference.

🚀 Deployment

Platform: Hugging Face Spaces

Framework: Gradio

Model format: model.pkl

Required Dependencies

Listed in requirements.txt:

scikit-learn
pandas
numpy
gradio

📁 Project Structure
├── app.py

├── model.pkl

├── requirements.txt

├── HR-Employee-Attrition.csv

├── README.md

🛠 Technologies Used

Python

Pandas, NumPy

Scikit-learn

Gradio

Hugging Face Spaces

✅ Conclusion

This project demonstrates a complete end-to-end ML workflow:

- Data preprocessing
- Model building
- Evaluation
- Serialization
- Deployment

🫡 If you have suggestions or identify a better approach, I would be happy to discuss and improve this project.

👨‍💻 Author

Riyad Khandaker
Machine Learning Enthusiast
