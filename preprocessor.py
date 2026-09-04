import pandas as pd
import numpy as np

# Encoding mappings for categorical features
GENDER_MAP = {'Male': 1, 'Female': 0}
OVERTIME_MAP = {'Yes': 1, 'No': 0}
MARITAL_MAP = {'Single': 0, 'Married': 1, 'Divorced': 2}
EDUCATION_MAP = {
    'High School': 1,
    'Associate Degree': 2,
    "Bachelor's Degree": 3,
    "Master's Degree": 4,
    'PhD': 5
}
DEPARTMENT_MAP = {
    'IT': 0,
    'Sales': 1,
    'Human Resources': 2,
    'Finance': 3,
    'Research & Development': 4
}
ROLE_MAP = {
    'Developer': 0,
    'Systems Analyst': 1,
    'Sales Executive': 2,
    'Sales Representative': 3,
    'HR Specialist': 4,
    'HR Manager': 5,
    'Financial Analyst': 6,
    'Accountant': 7,
    'Research Scientist': 8,
    'Laboratory Technician': 9,
    'Manager': 10
}

# Features order for the ML model
FEATURE_COLUMNS = [
    'age', 'gender', 'department', 'job_role', 'salary', 'monthly_income',
    'years_at_company', 'education', 'marital_status', 'work_life_balance',
    'job_satisfaction', 'performance_rating', 'overtime', 'distance_from_home'
]

def preprocess_single(employee_dict):
    """
    Preprocesses a single employee dictionary and returns a 2D numpy array for prediction.
    """
    # Create copy to avoid mutating original dictionary
    data = dict(employee_dict)
    
    # Map fields with defaults if not matched
    data['gender'] = GENDER_MAP.get(data.get('gender'), 1) # default Male
    data['overtime'] = OVERTIME_MAP.get(data.get('overtime'), 0) # default No
    data['marital_status'] = MARITAL_MAP.get(data.get('marital_status'), 1) # default Married
    data['education'] = EDUCATION_MAP.get(data.get('education'), 3) # default Bachelor
    data['department'] = DEPARTMENT_MAP.get(data.get('department'), 0) # default IT
    data['job_role'] = ROLE_MAP.get(data.get('job_role'), 0) # default Developer
    
    # Handle numerical types, with fallbacks
    try:
        data['age'] = int(data.get('age', 35))
        data['salary'] = float(data.get('salary', 50000))
        data['monthly_income'] = float(data.get('monthly_income', 4000))
        data['years_at_company'] = int(data.get('years_at_company', 5))
        data['work_life_balance'] = int(data.get('work_life_balance', 3))
        data['job_satisfaction'] = int(data.get('job_satisfaction', 3))
        data['performance_rating'] = int(data.get('performance_rating', 3))
        data['distance_from_home'] = float(data.get('distance_from_home', 5.0))
    except (ValueError, TypeError):
        data['age'] = 35
        data['salary'] = 50000
        data['monthly_income'] = 4000
        data['years_at_company'] = 5
        data['work_life_balance'] = 3
        data['job_satisfaction'] = 3
        data['performance_rating'] = 3
        data['distance_from_home'] = 5.0

    # Build feature array in the correct order
    features = [data[col] for col in FEATURE_COLUMNS]
    return np.array([features])

def preprocess_dataframe(df):
    """
    Preprocesses a pandas DataFrame of raw employee records.
    """
    processed_df = df.copy()
    processed_df['gender'] = processed_df['gender'].map(GENDER_MAP).fillna(1).astype(int)
    processed_df['overtime'] = processed_df['overtime'].map(OVERTIME_MAP).fillna(0).astype(int)
    processed_df['marital_status'] = processed_df['marital_status'].map(MARITAL_MAP).fillna(1).astype(int)
    processed_df['education'] = processed_df['education'].map(EDUCATION_MAP).fillna(3).astype(int)
    processed_df['department'] = processed_df['department'].map(DEPARTMENT_MAP).fillna(0).astype(int)
    processed_df['job_role'] = processed_df['job_role'].map(ROLE_MAP).fillna(0).astype(int)
    
    # Cast numericals
    for col in ['age', 'years_at_company', 'work_life_balance', 'job_satisfaction', 'performance_rating']:
        processed_df[col] = processed_df[col].fillna(3).astype(int)
    for col in ['salary', 'monthly_income', 'distance_from_home']:
        processed_df[col] = processed_df[col].fillna(0.0).astype(float)
        
    return processed_df[FEATURE_COLUMNS]
