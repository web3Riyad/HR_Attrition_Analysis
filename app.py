import numpy as np
import pandas as pd 
import pickle
import gradio as gr  

# 1. Load the Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_attrition(Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education, EducationField, EmployeeCount,
        EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate,
        JobInvolvement, JobLevel, JobRole, JobSatisfaction,
        MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
        OverTime, PercentSalaryHike, PerformanceRating,
        RelationshipSatisfaction, StandardHours, StockOptionLevel,
        TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
        YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
        YearsWithCurrManager):
    
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        Age,BusinessTravel,DailyRate,Department,DistanceFromHome,Education, EducationField, EmployeeCount,
        EmployeeNumber, EnvironmentSatisfaction, Gender, HourlyRate,
        JobInvolvement, JobLevel, JobRole, JobSatisfaction,
        MaritalStatus, MonthlyIncome, MonthlyRate, NumCompaniesWorked,
        OverTime, PercentSalaryHike, PerformanceRating,
        RelationshipSatisfaction, StandardHours, StockOptionLevel,
        TotalWorkingYears, TrainingTimesLastYear, WorkLifeBalance,
        YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion,
        YearsWithCurrManager

    ]],
    columns=[
        'Age', 'BusinessTravel', 'DailyRate', 'Department',
        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
        ])
    
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return formatted result (Clipped 0-5)
    return f"Predicted Attrition: {'Yes' if prediction == 1 else 'No'}"



# 3. The App Interface
inputs = [
    gr.Number(label="Age", value=18),
    gr.Dropdown( ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'],label = 'Business Travel'),
    gr.Number(value = 100, label = "Daily Rate"),
    gr.Dropdown( ['Sales' ,'Research & Development', 'Human Resources'],label = 'Department'),
    gr.Number(label = "Distance from Home(k.m)", value = 10),
    gr.Slider(0,5, step = 1,label = "Education"),
    gr.Dropdown(['Life Sciences', 'Other', 'Medical' ,'Marketing', 'Technical Degree',
'Human Resources'],label = "Education Field"),
    gr.Number(label = "EmployeeCount"),
    gr.Number(label = "EmployeeNumber"),
    gr.Slider(0,4,step = 1,label = "EnvironmentSatisfaction"),
    gr.Radio( ['Female' ,'Male'],label = "Gender"),
    gr.Number(value = 20,label = "HourlyRate"),
    gr.Slider(0,4,step =1,label = "JobInvolvement"),
    gr.Slider(0,5,step =1,label = "JobLevel"),
    gr.Dropdown( ['Sales Executive', 'Research Scientist', 'Laboratory Technician',
    'Manufacturing Director' ,'Healthcare Representative' ,'Manager',
    'Sales Representative', 'Research Director', 'Human Resources'],label = "JobRole"),
    gr.Slider(0,4,step =1,label = "JobSatisfaction"),
    gr.Dropdown(['Single' ,'Married' ,'Divorced'],label = "MaritalStatus"),
    gr.Number(value = 7500,label = "MonthlyIncome"),
    gr.Number(value = 15000,label = "MonthlyRate"),
    gr.Number(value = 2,label = "NumCompaniesWorked"),
    gr.Radio(["Yes","No"],label = "OverTime"),
    gr.Number(value = 15,label = "PercentSalaryHike"),
    gr.Number(value = 3,label = "PerformanceRating"),
    gr.Slider(1,4,step=1,label = "RelationshipSatisfaction"),
    gr.Number(value = 80,label = "StandardHours"),
    gr.Slider(0,3,step = 1,label = "StockOptionLevel"),
    gr.Number(value = 8,label = "TotalWorkingYears"),
    gr.Number(value = 3,label = "TrainingTimesLastYear"),
    gr.Slider(1,4,step = 1,label = "WorkLifeBalance"),
    gr.Number(value = 5,label = "YearsAtCompany"),
    gr.Number(value = 2.5,label = "YearsInCurrentRole"),
    gr.Number(value = 2,label = "YearsSinceLastPromotion"),
    gr.Number(value = 5,label = "YearsWithCurrManager"),
]
    
    
app = gr.Interface(
    fn=predict_attrition,
    inputs=inputs,
    outputs="text", 
    title="Employee Attrition prediction")

app.launch(share=True)
    