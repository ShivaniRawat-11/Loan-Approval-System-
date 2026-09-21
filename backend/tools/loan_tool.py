import os
import threading
import joblib
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from utils.helpers import get_logger

logger = get_logger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "ml_model")

# ── Thread-safe artifact loading ─────────────────────────────────────────────
_artifacts_lock = threading.Lock()
_artifacts = {
    "model": None,
    "scaler": None,
    "le_education": None,
    "ohe": None,
    "feature_names": None,
    "loaded": False,
}


def _load_artifacts():
    """Load ML model artifacts in a thread-safe manner. Only loads once."""
    with _artifacts_lock:
        if _artifacts["loaded"]:
            return True
        try:
            _artifacts["model"] = joblib.load(os.path.join(MODEL_DIR, "loan_model.pkl"))
            _artifacts["scaler"] = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
            _artifacts["le_education"] = joblib.load(os.path.join(MODEL_DIR, "le_education.pkl"))
            _artifacts["ohe"] = joblib.load(os.path.join(MODEL_DIR, "ohe.pkl"))
            _artifacts["feature_names"] = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
            _artifacts["loaded"] = True
            logger.info("ML model artifacts loaded successfully.")
            return True
        except FileNotFoundError as e:
            logger.error(f"ML model file not found: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading ML artifacts: {e}")
            return False


# ── Input Validation ─────────────────────────────────────────────────────────

def _validate_inputs(
    applicant_income, loan_amount, credit_score, age, loan_term,
    coapplicant_income, dependents, existing_loans, savings, collateral_value,
    credit_history=None
):
    """
    Validates loan prediction inputs against bank policy constraints.
    Returns (is_valid, error_message).
    """
    errors = []

    if applicant_income < 0:
        errors.append("Applicant income cannot be negative.")
    if coapplicant_income < 0:
        errors.append("Coapplicant income cannot be negative.")
    if loan_amount <= 0:
        errors.append("Loan amount must be greater than zero.")

    # Check credit score and credit history
    has_score = (credit_score is not None and credit_score >= 300 and credit_score <= 900)
    has_history = (credit_history is not None and credit_history in [0, 1, 0.0, 1.0])

    if not has_score and not has_history:
        errors.append("Please provide a valid credit score (300-900) or credit history (0 or 1).")
    elif has_score:
        if credit_score < 300 or credit_score > 900:
            errors.append(f"Credit score {credit_score} is outside valid range (300-900).")

    if age < 21 or age > 65:
        errors.append(f"Age {age} is outside eligible range (21-65 per bank policy).")
    if loan_term <= 0:
        errors.append("Loan term must be greater than zero.")
    if dependents < 0:
        errors.append("Number of dependents cannot be negative.")
    if existing_loans < 0:
        errors.append("Number of existing loans cannot be negative.")
    if savings < 0:
        errors.append("Savings cannot be negative.")
    if collateral_value < 0:
        errors.append("Collateral value cannot be negative.")

    if errors:
        return False, " | ".join(errors)
    return True, ""


class LoanInput(BaseModel):
    """Inputs for loan prediction. Strings are case-insensitive."""
    gender: str = Field(description="Gender (Male, Female)", default="Male")
    married: str = Field(description="Marital Status (Yes, No, Married, Single)", default="No")
    education: str = Field(description="Education Level (Graduate, Not Graduate)", default="Graduate")
    employment_status: str = Field(description="Employment Status (Salaried, Self-employed, Contract, Unemployed)", default="Salaried")
    applicant_income: float = Field(description="Applicant Income (monthly)")
    coapplicant_income: float = Field(description="Coapplicant Income (monthly)", default=0.0)
    loan_amount: float = Field(description="Loan Amount")
    loan_term: float = Field(description="Loan Term (in months)", default=36.0)
    credit_score: float = Field(description="Credit Score (e.g. 300-900)", default=0.0)
    property_area: str = Field(description="Property Area (Urban, Semiurban, Rural)", default="Semiurban")
    loan_purpose: str = Field(description="Loan Purpose (Home, Car, Personal, Education, Business)", default="Personal")
    employer_category: str = Field(description="Employer Category (Govt, Private, Business, MNC)", default="Private")
    age: float = Field(description="Applicant Age in years", default=30.0)
    dependents: float = Field(description="Number of dependents", default=0.0)
    existing_loans: float = Field(description="Number of existing loans", default=0.0)
    savings: float = Field(description="Total savings amount", default=0.0)
    collateral_value: float = Field(description="Collateral value for the loan", default=0.0)
    credit_history: float = Field(description="Credit History (1 for good, 0 for bad)", default=None)

@tool("predict_loan_approval", args_schema=LoanInput)
def predict_loan_approval(
    gender: str = "Male",
    married: str = "No",
    education: str = "Graduate",
    employment_status: str = "Salaried",
    applicant_income: float = 0.0,
    coapplicant_income: float = 0.0,
    loan_amount: float = 0.0,
    loan_term: float = 36.0,
    credit_score: float = 0.0,
    property_area: str = "Semiurban",
    loan_purpose: str = "Personal",
    employer_category: str = "Private",
    age: float = 30.0,
    dependents: float = 0.0,
    existing_loans: float = 0.0,
    savings: float = 0.0,
    collateral_value: float = 0.0,
    credit_history: float = None
) -> str:
    """Predict loan approval based on applicant features. Returns prediction and confidence."""

    # ── Step 0: Resolve credit score and history mapping ──────────────────
    if (credit_score < 300 or credit_score > 900) and (credit_history is not None and credit_history in [0, 1, 0.0, 1.0]):
        if credit_history == 1 or credit_history == 1.0:
            credit_score = 750.0
        else:
            credit_score = 550.0
    elif (credit_score >= 300 and credit_score <= 900) and (credit_history is None):
        credit_history = 1.0 if credit_score >= 650 else 0.0

    # ── Step 0.1: Validate inputs ──────────────────────────────────────────
    is_valid, validation_error = _validate_inputs(
        applicant_income, loan_amount, credit_score, age, loan_term,
        coapplicant_income, dependents, existing_loans, savings, collateral_value,
        credit_history
    )
    if not is_valid:
        logger.warning(f"Input validation failed: {validation_error}")
        return f"Input validation error: {validation_error}"

    # ── Step 1: Load model artifacts ─────────────────────────────────────
    if not _artifacts["loaded"]:
        if not _load_artifacts():
            return "Error: ML model files not found. Please ensure model files exist in the 'ml_model' directory."

    model = _artifacts["model"]
    scaler = _artifacts["scaler"]
    le_education = _artifacts["le_education"]
    ohe = _artifacts["ohe"]
    feature_names = _artifacts["feature_names"]

    try:
        # ── Step 2: Normalize string inputs ──────────────────────────────
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
            emp_val = "Salaried"

        # ── Step 3: Compute derived features ─────────────────────────────
        base_rate = 0.085
        if credit_score >= 750:
            rate = base_rate
        elif credit_score >= 700:
            rate = base_rate + 0.01
        elif credit_score >= 650:
            rate = base_rate + 0.025
        else:
            rate = base_rate + 0.05

        r = rate / 12
        n = loan_term if loan_term > 0 else 1
        
        if r > 0:
            emi = (loan_amount * r * (1 + r)**n) / ((1 + r)**n - 1)
        else:
            emi = loan_amount / n

        total_income = applicant_income + coapplicant_income
        if total_income > 0:
            dti_ratio = emi / total_income
        else:
            dti_ratio = 1.0

        logger.debug(
            f"Prediction input: income={applicant_income}, loan={loan_amount}, "
            f"credit={credit_score}, dti={dti_ratio:.4f}, age={age}"
        )

        # ── Step 4: Build feature DataFrame ──────────────────────────────
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
            "Collateral_Value": [collateral_value],
        }

        df_input = pd.DataFrame(data)

        # ── Step 5: Encode features ──────────────────────────────────────
        try:
            df_input["Education_Level"] = le_education.transform(df_input["Education_Level"])
        except (ValueError, KeyError) as e:
            logger.warning(f"Education label encoding fallback: {e}")
            df_input["Education_Level"] = 0

        ohe_cols = [
            "Employment_Status",
            "Marital_Status",
            "Loan_Purpose",
            "Property_Area",
            "Gender",
            "Employer_Category",
        ]

        try:
            encoded_array = ohe.transform(df_input[ohe_cols])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=ohe.get_feature_names_out(ohe_cols),
                index=df_input.index,
            )
        except Exception as e:
            logger.error(f"OneHot encoding failed: {e}")
            return f"Error: Could not encode input features. {str(e)}"

        df_remaining = df_input.drop(columns=ohe_cols)
        df_processed = pd.concat([df_remaining, encoded_df], axis=1)

        final_input = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            if col in df_processed.columns:
                final_input[col] = df_processed[col]
            else:
                final_input[col] = 0

        # ── Step 6: Scale and predict ────────────────────────────────────
        final_input_scaled = scaler.transform(final_input)
        prediction = model.predict(final_input_scaled)[0]

        try:
            proba = model.predict_proba(final_input_scaled)[0]
            confidence = max(proba) * 100
        except AttributeError:
            logger.warning("Model does not support predict_proba, confidence unavailable.")
            confidence = 0.0

        status = "APPROVED" if prediction == 1 else "REJECTED"
        logger.info(f"Prediction result: {status} (confidence: {confidence:.1f}%)")

        # ── Step 7: Generate feedback ────────────────────────────────────
        feedback = []
        if prediction == 0:
            if credit_score < 650:
                feedback.append(f"Credit score {credit_score:.0f} is below minimum (650 required per policy).")
            elif credit_score < 700:
                feedback.append(f"Credit score {credit_score:.0f} is marginal (above 700 recommended).")
            if total_income < 2500:
                feedback.append(f"Total monthly income ${total_income:,.0f} is below minimum ($2,500 required).")
            if dti_ratio > 0.43:
                feedback.append(f"Debt-to-income ratio {dti_ratio:.1%} exceeds 43% policy limit.")
            if emp_val == "Unemployed":
                feedback.append("Unemployed applicants are not eligible per policy.")
            if not feedback:
                feedback.append("Consider improving credit score, reducing existing debt, or increasing income.")

        feedback_str = " ".join(feedback) if feedback else "Congratulations! Your profile meets the requirements."

        return (
            f"Loan Status: {status} (Confidence: {confidence:.1f}%)\n"
            f"Details: {feedback_str}"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return f"Error during prediction: {str(e)}"
