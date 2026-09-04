import os
import joblib
import numpy as np
import preprocessor

# Global variable to cache the loaded model payload
_MODEL_PAYLOAD = None

def load_model():
    """
    Loads the trained model payload from model.joblib.
    """
    global _MODEL_PAYLOAD
    if _MODEL_PAYLOAD is None:
        model_path = os.path.join(os.path.dirname(__file__), 'model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Please run 'python train_model.py' to train and save the model first."
            )
        _MODEL_PAYLOAD = joblib.load(model_path)
    return _MODEL_PAYLOAD

def predict_attrition(employee_dict):
    """
    Predicts attrition probability, determines risk level, and generates tailored recommendations.
    
    Parameters:
        employee_dict (dict): A dictionary of raw employee features.
        
    Returns:
        dict: {
            'prediction': 'Likely to Leave' or 'Likely to Stay',
            'leave_probability': float (0-100),
            'stay_probability': float (0-100),
            'risk_level': 'Low' | 'Medium' | 'High',
            'recommendations': str (semicolon or newline-separated recommendation text),
            'risk_factors': list of str
        }
    """
    payload = load_model()
    model = payload['model']
    
    # Preprocess single record to numpy array
    X_single = preprocessor.preprocess_single(employee_dict)
    
    # Predict probabilities
    # Classes: 0 = Stay, 1 = Leave
    probs = model.predict_proba(X_single)[0]
    leave_prob = float(probs[1]) * 100.0
    stay_prob = float(probs[0]) * 100.0
    
    # Determine binary prediction
    # If model.predict() is preferred, we use threshold 50%
    pred_class = int(model.predict(X_single)[0])
    prediction = "Likely to Leave" if pred_class == 1 or leave_prob >= 50.0 else "Likely to Stay"
    
    # Risk Level mapping
    # 0–30% -> Low
    # 31–70% -> Medium
    # 71–100% -> High
    if leave_prob <= 30.0:
        risk_level = "Low"
    elif leave_prob <= 70.0:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    # Analyze risk factors and compile customized recommendations
    risk_factors = []
    recs = []
    
    overtime = str(employee_dict.get('overtime', 'No')).strip().lower()
    job_satisfaction = int(employee_dict.get('job_satisfaction', 3))
    work_life_balance = int(employee_dict.get('work_life_balance', 3))
    distance = float(employee_dict.get('distance_from_home', 5.0))
    monthly_income = float(employee_dict.get('monthly_income', 4000.0))
    salary = float(employee_dict.get('salary', 50000.0))
    years_at_company = int(employee_dict.get('years_at_company', 5))
    performance_rating = int(employee_dict.get('performance_rating', 3))
    
    # Check factors and generate actionable feedback
    if overtime == 'yes':
        risk_factors.append("High Overtime Workload")
        recs.append("Review workload distribution and reduce mandatory overtime. Consider expanding resources or team capacity.")
        
    if job_satisfaction <= 2:
        risk_factors.append("Low Job Satisfaction")
        recs.append("Conduct an informal HR or manager-led 1-on-1 feedback discussion to identify friction points, role alignment, or workspace issues.")
        
    if work_life_balance <= 2:
        risk_factors.append("Poor Work-Life Balance")
        recs.append("Introduce flexible hours, remote/hybrid scheduling options, or set clearer boundaries on off-hours communications.")
        
    if distance >= 15.0:
        risk_factors.append("Long Commute Distance")
        recs.append("Provide commuter benefits, support for hybrid remote work schedules, or explore nearby satellite office spaces.")
        
    if monthly_income < 4000.0:
        risk_factors.append("Low Compensation Bracket")
        recs.append("Perform a compensation benchmark audit. Adjust base salary/allowances closer to industry averages for this role.")
        
    if years_at_company <= 2 and years_at_company > 0:
        risk_factors.append("Early Career/Tenure Churn Risk")
        recs.append("Enroll in new-hire mentorship circles and map out clear, tangible milestones for internal career growth within 12-24 months.")
        
    if performance_rating >= 4 and (job_satisfaction <= 2 or monthly_income < 6000.0):
        risk_factors.append("Retention Threat to High Performer")
        recs.append("High Performer with low satisfaction/compensation! Prioritize immediately for retention bonuses, fast-track promotion pathways, or specialized training.")

    # Default recommendation if risk is low and no factors trigger
    if not recs:
        if risk_level == "Low":
            recs.append("Maintain current engagement levels. Schedule routine annual development reviews and recognize achievements.")
        else:
            recs.append("Conduct a standard pulse check to verify role satisfaction and general well-being.")

    # Format recommendations as bullet points
    recommendations_text = "\n".join([f"• {r}" for r in recs])
    
    return {
        'prediction': prediction,
        'leave_probability': round(leave_prob, 1),
        'stay_probability': round(stay_prob, 1),
        'risk_level': risk_level,
        'recommendations': recommendations_text,
        'risk_factors': risk_factors
    }
