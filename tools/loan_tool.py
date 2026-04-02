
import os
import joblib
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ML model")

# Global variables for artifacts
model = None
scaler = None
le_education = None
ohe = None
feature_names = None

def load_artifacts():
    global model, scaler, le_education, ohe, feature_names
    try:
        model = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        le_education = joblib.load(os.path.join(MODEL_DIR, "le_education.pkl"))
        ohe = joblib.load(os.path.join(MODEL_DIR, "ohe.pkl"))
        feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
        return True
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return False

class LoanInput(BaseModel):
    """Inputs for loan prediction. Strings are case-insensitive."""
    gender: str = Field(description="Gender (Male, Female)", default="Male")
    married: str = Field(description="Marital Status (Yes, No, Married, Single)", default="No")
    education: str = Field(description="Education Level (Graduate, Not Graduate)", default="Graduate")
    employment_status: str = Field(description="Employment Status (Salaried, Self-employed, Contract, Unemployed)", default="Salaried")
    applicant_income: float = Field(description="Applicant Income (monthly)")
    coapplicant_income: float = Field(description="Coapplicant Income (monthly)", default=0.0)
    loan_amount: float = Field(description="Loan Amount")
    loan_term: float = Field(description="Loan Term (in months)", default=360.0)
    credit_score: float = Field(description="Credit Score (e.g. 300-900)")
    property_area: str = Field(description="Property Area (Urban, Semiurban, Rural)", default="Semiurban")
    loan_purpose: str = Field(description="Loan Purpose (Home, Car, Personal, Education, Business)", default="Personal")
    employer_category: str = Field(description="Employer Category (Govt, Private, Business, MNC)", default="Private")
    age: float = Field(description="Applicant Age in years", default=30.0)
    dependents: float = Field(description="Number of dependents", default=0.0)
    existing_loans: float = Field(description="Number of existing loans", default=0.0)
    savings: float = Field(description="Total savings amount", default=0.0)
    collateral_value: float = Field(description="Collateral value for the loan", default=0.0)

@tool("predict_loan_approval", args_schema=LoanInput)
def predict_loan_approval(
    gender: str = "Male",
    married: str = "No",
    education: str = "Graduate",
    employment_status: str = "Salaried",
    applicant_income: float = 0.0,
    coapplicant_income: float = 0.0,
    loan_amount: float = 0.0,
    loan_term: float = 360.0,
    credit_score: float = 0.0,
    property_area: str = "Semiurban",
    loan_purpose: str = "Personal",
    employer_category: str = "Private",
    age: float = 30.0,
    dependents: float = 0.0,
    existing_loans: float = 0.0,
    savings: float = 0.0,
    collateral_value: float = 0.0
) -> str:
    """Predict loan approval based on applicant features. Returns prediction and confidence."""
    
    global model, scaler, le_education, ohe, feature_names
    
    if model is None:
        load_artifacts()
        if model is None:
            return "Error: ML model files not found."

    try:
        # Normalize Inputs
        gender_map = {"male": "Male", "female": "Female"}
        
        married_lower = married.lower()
        if "couple" in married_lower or "married" in married_lower or "yes" in married_lower:
            married_val = "Married"
        else:
            married_val = "Single"
            
        edu_lower = education.lower()
        if "not" in edu_lower:
            edu_val = "Not Graduate"
        else:
            edu_val = "Graduate"
            
        emp_lower = employment_status.lower()
        if "self" in emp_lower:
            emp_val = "Self-employed"
        elif "unemployed" in emp_lower or "no" in emp_lower:
            emp_val = "Unemployed"
        elif "contract" in emp_lower:
            emp_val = "Contract"
        else:
            emp_val = "Salaried" # Default for "yes", "employed"

        # Compute sensible defaults for missing fields
        # If savings not provided, estimate as 6 months of income
        if savings <= 0:
            savings = applicant_income * 6
        # If collateral not provided, estimate as 1.2x loan amount
        if collateral_value <= 0:
            collateral_value = loan_amount * 1.2
        # Compute DTI ratio (Debt-to-Income)
        total_income = applicant_income + coapplicant_income
        dti_ratio = (loan_amount / (total_income * loan_term)) if total_income > 0 else 1.0

        # Prepare input dict - column names MUST match model's feature_names
        data = {
            "Gender": [gender_map.get(gender.lower(), "Male")],
            "Marital_Status": [married_val],
            "Education_Level": [edu_val],
            "Employment_Status": [emp_val],
            "Applicant_Income": [applicant_income],
            "Coapplicant_Income": [coapplicant_income],
            "Loan_Amount": [loan_amount],
            "Loan_Term": [loan_term],
            "Credit_Score": [credit_score],
            "Property_Area": [property_area.title()],
            "Loan_Purpose": [loan_purpose.title()],
            "Employer_Category": [employer_category.title()],
            "Age": [age],
            "Dependents": [dependents],
            "Existing_Loans": [existing_loans],
            "DTI_Ratio": [dti_ratio],
            "Savings": [savings],
            "Collateral_Value": [collateral_value]
        }
        
        # Create DataFrame
        df_input = pd.DataFrame(data)
        
        # 1. Label Encode Education
        try:
            df_input["Education_Level"] = le_education.transform(df_input["Education_Level"])
        except:
             # Fallback
            df_input["Education_Level"] = 0
                
        # 2. OneHot Encode Categorical Cols
        ohe_cols = [
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category"
        ]
        
        encoded_array = ohe.transform(df_input[ohe_cols])
        encoded_df = pd.DataFrame(
            encoded_array, 
            columns=ohe.get_feature_names_out(ohe_cols),
            index=df_input.index
        )
        
        # 3. Combine
        df_remaining = df_input.drop(columns=ohe_cols) 
        df_processed = pd.concat([df_remaining, encoded_df], axis=1)
        
        # 4. Reorder columns
        final_input = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            if col in df_processed.columns:
                final_input[col] = df_processed[col]
            else:
                final_input[col] = 0
        
        # 5. Scale
        final_input_scaled = scaler.transform(final_input)
        
        # 6. Predict
        prediction = model.predict(final_input_scaled)[0]
        
        try:
            proba = model.predict_proba(final_input_scaled)[0]
            confidence = max(proba) * 100
        except:
            confidence = 0.0
            
        status = "APPROVED" if prediction == 1 else "REJECTED"
        
        # Basic feedback logic
        feedback = []
        if prediction == 0:
            if credit_score < 700:
                feedback.append(f"Credit score {credit_score} is low (recommended > 700).")
            if (applicant_income + coapplicant_income) < 4000:
                feedback.append("Total income might be too low.")
            if emp_val == "Unemployed":
                feedback.append("Employment status is Unemployed.")
                
        feedback_str = " ".join(feedback) if feedback else "Consider improving credit score or income."
        
        return f"Loan Status: {status} (Confidence: {confidence:.1f}%)\nDetails: {feedback_str if prediction == 0 else 'Congratulations!'}"

    except Exception as e:
        import traceback
        return f"Error during prediction: {str(e)}"
