import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import preprocessor

def generate_synthetic_data(num_samples=1500, random_seed=42):
    np.random.seed(random_seed)
    
    # Pre-defined category sets
    genders = ['Male', 'Female']
    departments = ['IT', 'Sales', 'Human Resources', 'Finance', 'Research & Development']
    
    dept_roles = {
        'IT': ['Developer', 'Systems Analyst', 'Manager'],
        'Sales': ['Sales Executive', 'Sales Representative', 'Manager'],
        'Human Resources': ['HR Specialist', 'HR Manager', 'Manager'],
        'Finance': ['Financial Analyst', 'Accountant', 'Manager'],
        'Research & Development': ['Research Scientist', 'Laboratory Technician', 'Manager']
    }
    
    educations = ['High School', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'PhD']
    maritals = ['Single', 'Married', 'Divorced']
    overtimes = ['Yes', 'No']
    
    data = []
    for i in range(num_samples):
        # Base demographics
        age = int(np.random.randint(20, 60))
        gender = np.random.choice(genders)
        marital = np.random.choice(maritals, p=[0.3, 0.5, 0.2])
        education = np.random.choice(educations, p=[0.1, 0.2, 0.5, 0.15, 0.05])
        
        # Job features
        dept = np.random.choice(departments, p=[0.3, 0.3, 0.1, 0.1, 0.2])
        job_role = np.random.choice(dept_roles[dept])
        
        # Years at company aligned with age
        max_years = min(age - 18, 25)
        years_at_company = int(np.random.randint(0, max_years + 1)) if max_years > 0 else 0
        
        # Distance from home (km)
        distance = round(float(np.random.uniform(1.0, 30.0)), 1)
        
        # Overtime
        overtime = np.random.choice(overtimes, p=[0.28, 0.72])
        
        # Ratings and satisfaction scales (1-4)
        # Seeded with slight positive bias
        job_satisfaction = int(np.random.choice([1, 2, 3, 4], p=[0.15, 0.2, 0.4, 0.25]))
        work_life_balance = int(np.random.choice([1, 2, 3, 4], p=[0.1, 0.25, 0.45, 0.2]))
        performance_rating = int(np.random.choice([1, 2, 3, 4], p=[0.05, 0.15, 0.6, 0.2]))
        
        # Income and Salary details based on job role and age
        # base income for roles
        role_base_income = {
            'Developer': 5000,
            'Systems Analyst': 6000,
            'Sales Executive': 5500,
            'Sales Representative': 3500,
            'HR Specialist': 4500,
            'HR Manager': 8000,
            'Financial Analyst': 5800,
            'Accountant': 5200,
            'Research Scientist': 6200,
            'Laboratory Technician': 4000,
            'Manager': 10000
        }
        
        base_inc = role_base_income.get(job_role, 5000)
        # Experience increment
        inc_exp = years_at_company * 250
        # Age increment
        inc_age = (age - 20) * 100
        # Add random noise
        monthly_income = base_inc + inc_exp + inc_age + np.random.randint(-500, 500)
        monthly_income = max(2000.0, monthly_income)
        
        # Annual salary is monthly * 12
        salary = monthly_income * 12
        
        # Calculate attrition probability based on rules:
        # Attrition is more likely if:
        # 1. High overtime (+0.25)
        # 2. Low Job satisfaction (+0.20 for 1, +0.10 for 2)
        # 3. Poor Work-Life balance (+0.15 for 1, +0.05 for 2)
        # 4. Long distance from home (+0.10 if distance > 18)
        # 5. Low monthly income relative to age (+0.15 if monthly_income < 4000)
        # 6. Single marital status (+0.08)
        # 7. Short tenure (+0.10 if years_at_company <= 2)
        
        prob = 0.05 # base prob
        
        if overtime == 'Yes':
            prob += 0.25
        if job_satisfaction == 1:
            prob += 0.22
        elif job_satisfaction == 2:
            prob += 0.10
            
        if work_life_balance == 1:
            prob += 0.18
        elif work_life_balance == 2:
            prob += 0.06
            
        if distance > 18.0:
            prob += 0.12
            
        if monthly_income < 4000:
            prob += 0.15
        elif monthly_income > 9000:
            prob -= 0.10
            
        if marital == 'Single':
            prob += 0.08
            
        if years_at_company <= 2:
            prob += 0.12
            
        # Bound probability between 0.02 and 0.98
        prob = max(0.02, min(0.98, prob))
        
        # Generate target outcome with high determinism to ensure model has high responsiveness
        if prob >= 0.30:
            attrition = 1 if np.random.rand() < 0.95 else 0
        else:
            attrition = 1 if np.random.rand() < 0.05 else 0
        
        data.append({
            'employee_id': f"EMP{1000+i}",
            'name': f"Employee_{i+1}",
            'age': age,
            'gender': gender,
            'department': dept,
            'job_role': job_role,
            'salary': salary,
            'monthly_income': monthly_income,
            'years_at_company': years_at_company,
            'education': education,
            'marital_status': marital,
            'work_life_balance': work_life_balance,
            'job_satisfaction': job_satisfaction,
            'performance_rating': performance_rating,
            'overtime': overtime,
            'distance_from_home': distance,
            'attrition': attrition
        })
        
    return pd.DataFrame(data)

def main():
    print("Generating synthetic HR employee dataset...")
    raw_df = generate_synthetic_data(num_samples=1500)
    print(f"Dataset generated. Shape: {raw_df.shape}")
    print(f"Attrition count: {raw_df['attrition'].value_counts().to_dict()}")
    
    # Save the synthetic dataset to csv for record/reference
    raw_df.to_csv('employee_data_raw.csv', index=False)
    print("Saved raw synthetic data to 'employee_data_raw.csv'")
    
    # Preprocess
    print("Preprocessing features...")
    X = preprocessor.preprocess_dataframe(raw_df)
    y = raw_df['attrition']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training and Comparing Models ---")
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])
    print(f"Logistic Regression - Accuracy: {lr_acc:.4f}, ROC-AUC: {lr_auc:.4f}")
    
    # 2. Decision Tree
    dt = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
    print(f"Decision Tree - Accuracy: {dt_acc:.4f}, ROC-AUC: {dt_auc:.4f}")
    
    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"Random Forest - Accuracy: {rf_acc:.4f}, ROC-AUC: {rf_auc:.4f}")
    
    # Choose best model (RF by default or highest accuracy)
    best_model = rf
    best_name = "Random Forest"
    best_acc = rf_acc
    
    if dt_acc > best_acc and dt_acc > lr_acc:
        best_model = dt
        best_name = "Decision Tree"
        best_acc = dt_acc
    elif lr_acc > best_acc:
        best_model = lr
        best_name = "Logistic Regression"
        best_acc = lr_acc
        
    print(f"\nSelecting {best_name} as the production model (Accuracy: {best_acc:.4f}).")
    print("\nClassification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))
    
    # Feature Importances for RF
    if hasattr(best_model, 'feature_importances_'):
        print("\nFeature Importances:")
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        for f in range(X.shape[1]):
            print(f"{f + 1}. {X.columns[indices[f]]:<22} : {importances[indices[f]]:.4f}")
            
    # Serialize best model
    model_payload = {
        'model': best_model,
        'model_name': best_name,
        'features': list(X.columns),
        'mappings': {
            'gender': preprocessor.GENDER_MAP,
            'overtime': preprocessor.OVERTIME_MAP,
            'marital_status': preprocessor.MARITAL_MAP,
            'education': preprocessor.EDUCATION_MAP,
            'department': preprocessor.DEPARTMENT_MAP,
            'job_role': preprocessor.ROLE_MAP
        }
    }
    
    joblib.dump(model_payload, 'model.joblib')
    print("\nProduction model and preprocessor schema saved to 'model.joblib'")

if __name__ == '__main__':
    main()
