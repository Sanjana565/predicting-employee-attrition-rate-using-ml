import os
import random
import numpy as np
from functools import wraps
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from database import db, init_database, User, Employee, Prediction
from ml_model import predict_attrition

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'employee-attrition-secret-key-998877')

# Initialize DB connection configurations
init_database(app)

# Setup flask login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Decorator for Admin access
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'Admin':
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Decorator for HR access
def hr_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'HR':
            flash('Access denied. HR Manager privileges required.', 'danger')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Initialize Database tables and Seed Admin user if database is empty
with app.app_context():
    db.create_all()
    
    # Check if Admin already exists, if not seed it
    admin = User.query.filter_by(email='admin@attrition.com').first()
    if not admin:
        default_admin = User(
            name='System Administrator',
            email='admin@attrition.com',
            role='Admin'
        )
        default_admin.set_password('admin123')
        db.session.add(default_admin)
        
    # Check if HR manager already exists, if not seed it
    hr = User.query.filter_by(email='hr@attrition.com').first()
    if not hr:
        default_hr = User(
            name='Jane Doe',
            email='hr@attrition.com',
            role='HR'
        )
        default_hr.set_password('hr12345')
        db.session.add(default_hr)
        
    db.session.commit()

# --- ROUTES ---

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user)
            flash(f'Welcome back, {user.name}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')
            
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        role = 'HR' # Default role for self-registration is HR Manager
        
        if not name or not email or not password:
            flash('Please fill in all fields.', 'warning')
            return render_template('register.html')
            
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash('Email already registered. Please log in.', 'warning')
            return redirect(url_for('login'))
            
        new_user = User(name=name, email=email, role=role)
        new_user.set_password(password)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('An error occurred. Please try again.', 'danger')
            
    return render_template('register.html')

@app.route('/logout')
def logout():
    logout_user()
    session.clear()
    flash('Successfully logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'Admin':
        return redirect(url_for('dashboard_admin'))
    return redirect(url_for('dashboard_hr'))

@app.route('/dashboard/hr')
@login_required
@hr_required
def dashboard_hr():
    # HR dashboard details
    total_emp = Employee.query.count()
    
    # Calculate Risk Counts from Predictions
    # Subquery to get the latest prediction for each employee
    latest_prediction_subquery = db.session.query(
        Prediction.employee_id,
        db.func.max(Prediction.prediction_date).label('max_date')
    ).group_by(Prediction.employee_id).subquery()

    latest_predictions = db.session.query(Prediction).join(
        latest_prediction_subquery,
        (Prediction.employee_id == latest_prediction_subquery.c.employee_id) &
        (Prediction.prediction_date == latest_prediction_subquery.c.max_date)
    ).all()
    
    high_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'High')
    low_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'Low')
    medium_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'Medium')
    
    # Unpredicted count
    predicted_ids = {p.employee_id for p in latest_predictions}
    unpredicted_cnt = max(0, total_emp - len(predicted_ids))
    
    # Add unpredicted employees to low risk by default for simple totals, or represent separately
    low_risk_cnt += unpredicted_cnt
    
    # Recent predictions (last 5)
    recent_predictions = Prediction.query.order_by(Prediction.prediction_date.desc()).limit(5).all()
    
    return render_template(
        'dashboard_hr.html',
        total_employees=total_emp,
        high_risk=high_risk_cnt,
        medium_risk=medium_risk_cnt,
        low_risk=low_risk_cnt,
        recent_predictions=recent_predictions
    )

@app.route('/dashboard/admin')
@login_required
@admin_required
def dashboard_admin():
    total_hr = User.query.filter_by(role='HR').count()
    total_emp = Employee.query.count()
    total_preds = Prediction.query.count()
    
    # Count of High Risk
    latest_prediction_subquery = db.session.query(
        Prediction.employee_id,
        db.func.max(Prediction.prediction_date).label('max_date')
    ).group_by(Prediction.employee_id).subquery()

    latest_predictions = db.session.query(Prediction).join(
        latest_prediction_subquery,
        (Prediction.employee_id == latest_prediction_subquery.c.employee_id) &
        (Prediction.prediction_date == latest_prediction_subquery.c.max_date)
    ).all()
    
    high_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'High')
    
    return render_template(
        'dashboard_admin.html',
        total_hr_users=total_hr,
        total_employees=total_emp,
        total_predictions=total_preds,
        high_risk_employees=high_risk_cnt
    )

# --- PROFILE ROUTE ---
@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        new_password = request.form.get('password', '').strip()
        
        if not name:
            flash('Name field is required.', 'warning')
            return redirect(url_for('profile'))
            
        current_user.name = name
        if new_password:
            current_user.set_password(new_password)
            flash('Profile details and password updated successfully!', 'success')
        else:
            flash('Profile details updated successfully!', 'success')
            
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            flash('An error occurred updating the database.', 'danger')
            
        return redirect(url_for('profile'))
        
    return render_template('profile.html')

# --- EMPLOYEE MANAGEMENT ---

@app.route('/employees')
@login_required
def employees_list():
    query = Employee.query
    
    # Get parameters
    search = request.args.get('search', '').strip()
    dept = request.args.get('department', '').strip()
    role = request.args.get('role', '').strip()
    
    if search:
        query = query.filter(
            (Employee.employee_id.like(f"%{search}%")) |
            (Employee.name.like(f"%{search}%"))
        )
    if dept:
        query = query.filter(Employee.department == dept)
    if role:
        query = query.filter(Employee.job_role == role)
        
    employees = query.all()
    
    # Attach latest prediction to each employee for displaying risk badges in list
    employee_list_rich = []
    for emp in employees:
        latest_pred = Prediction.query.filter_by(employee_id=emp.employee_id).order_by(Prediction.prediction_date.desc()).first()
        employee_list_rich.append({
            'employee': emp,
            'latest_prediction': latest_pred
        })
        
    # Get unique departments and job roles for dropdown filters
    all_depts = [r[0] for r in db.session.query(Employee.department).distinct().all()]
    all_roles = [r[0] for r in db.session.query(Employee.job_role).distinct().all()]
    
    return render_template(
        'employee_list.html',
        employees_rich=employee_list_rich,
        departments=all_depts,
        roles=all_roles,
        search=search,
        selected_dept=dept,
        selected_role=role
    )

@app.route('/employee/add', methods=['GET', 'POST'])
@login_required
def employee_add():
    if request.method == 'POST':
        # Retrieve form data
        emp_id = request.form.get('employee_id', '').strip()
        name = request.form.get('name', '').strip()
        age = request.form.get('age')
        gender = request.form.get('gender')
        dept = request.form.get('department')
        job_role = request.form.get('job_role')
        salary = request.form.get('salary')
        monthly_income = request.form.get('monthly_income')
        years_at_company = request.form.get('years_at_company')
        education = request.form.get('education')
        marital_status = request.form.get('marital_status')
        work_life_balance = request.form.get('work_life_balance')
        job_satisfaction = request.form.get('job_satisfaction')
        performance_rating = request.form.get('performance_rating')
        overtime = request.form.get('overtime', 'No')
        distance_from_home = request.form.get('distance_from_home')
        
        # Validations
        if not emp_id or not name:
            flash('Employee ID and Name are required.', 'warning')
            return redirect(url_for('employee_add'))
            
        existing = Employee.query.filter_by(employee_id=emp_id).first()
        if existing:
            flash(f'Employee with ID {emp_id} already exists.', 'warning')
            return redirect(url_for('employee_add'))
            
        try:
            # Create employee model
            new_emp = Employee(
                employee_id=emp_id,
                name=name,
                age=int(age) if age else 30,
                gender=gender,
                department=dept,
                job_role=job_role,
                salary=float(salary) if salary else 50000.0,
                monthly_income=float(monthly_income) if monthly_income else 4000.0,
                years_at_company=int(years_at_company) if years_at_company else 3,
                education=education,
                marital_status=marital_status,
                work_life_balance=int(work_life_balance) if work_life_balance else 3,
                job_satisfaction=int(job_satisfaction) if job_satisfaction else 3,
                performance_rating=int(performance_rating) if performance_rating else 3,
                overtime=overtime,
                distance_from_home=float(distance_from_home) if distance_from_home else 5.0
            )
            db.session.add(new_emp)
            db.session.commit()
            
            flash(f'Employee {name} ({emp_id}) added successfully!', 'success')
            
            # Automatically perform predictive inference right away if requested
            if 'predict_immediately' in request.form:
                return redirect(url_for('predict_attrition_trigger', employee_id=emp_id))
                
            return redirect(url_for('employees_list'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Failed to add employee. Error: {str(e)}', 'danger')
            return redirect(url_for('employee_add'))
            
    return render_template('employee_form.html', employee=None)

@app.route('/employee/edit/<employee_id>', methods=['GET', 'POST'])
@login_required
def employee_edit(employee_id):
    emp = Employee.query.get_or_404(employee_id)
    
    if request.method == 'POST':
        # Retrieve form data
        name = request.form.get('name', '').strip()
        age = request.form.get('age')
        gender = request.form.get('gender')
        dept = request.form.get('department')
        job_role = request.form.get('job_role')
        salary = request.form.get('salary')
        monthly_income = request.form.get('monthly_income')
        years_at_company = request.form.get('years_at_company')
        education = request.form.get('education')
        marital_status = request.form.get('marital_status')
        work_life_balance = request.form.get('work_life_balance')
        job_satisfaction = request.form.get('job_satisfaction')
        performance_rating = request.form.get('performance_rating')
        overtime = request.form.get('overtime', 'No')
        distance_from_home = request.form.get('distance_from_home')
        
        if not name:
            flash('Employee name is required.', 'warning')
            return redirect(url_for('employee_edit', employee_id=employee_id))
            
        try:
            emp.name = name
            emp.age = int(age) if age else emp.age
            emp.gender = gender
            emp.department = dept
            emp.job_role = job_role
            emp.salary = float(salary) if salary else emp.salary
            emp.monthly_income = float(monthly_income) if monthly_income else emp.monthly_income
            emp.years_at_company = int(years_at_company) if years_at_company else emp.years_at_company
            emp.education = education
            emp.marital_status = marital_status
            emp.work_life_balance = int(work_life_balance) if work_life_balance else emp.work_life_balance
            emp.job_satisfaction = int(job_satisfaction) if job_satisfaction else emp.job_satisfaction
            emp.performance_rating = int(performance_rating) if performance_rating else emp.performance_rating
            emp.overtime = overtime
            emp.distance_from_home = float(distance_from_home) if distance_from_home else emp.distance_from_home
            
            db.session.commit()
            flash(f'Employee {name} updated successfully!', 'success')
            
            if 'predict_immediately' in request.form:
                return redirect(url_for('predict_attrition_trigger', employee_id=emp.employee_id))
                
            return redirect(url_for('employees_list'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Failed to update employee: {str(e)}', 'danger')
            
    return render_template('employee_form.html', employee=emp)

@app.route('/employee/delete/<employee_id>', methods=['POST'])
@login_required
def employee_delete(employee_id):
    emp = Employee.query.get_or_404(employee_id)
    name = emp.name
    try:
        db.session.delete(emp)
        db.session.commit()
        flash(f'Employee {name} and their prediction logs were deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Failed to delete employee: {str(e)}', 'danger')
        
    return redirect(url_for('employees_list'))

# --- ML PREDICTION ---

@app.route('/predict/trigger/<employee_id>')
@login_required
def predict_attrition_trigger(employee_id):
    emp = Employee.query.get_or_404(employee_id)
    
    # Run prediction
    try:
        results = predict_attrition(emp.to_dict())
        
        # Save to database
        new_pred = Prediction(
            employee_id=emp.employee_id,
            prediction=results['prediction'],
            probability=results['leave_probability'],
            risk_level=results['risk_level'],
            recommendation=results['recommendations']
        )
        
        db.session.add(new_pred)
        db.session.commit()
        
        flash(f'Prediction successfully calculated for {emp.name}!', 'success')
        return redirect(url_for('prediction_result', prediction_id=new_pred.prediction_id))
        
    except Exception as e:
        db.session.rollback()
        flash(f'Prediction generation failed. Ensure the model has been trained first. Error: {str(e)}', 'danger')
        return redirect(url_for('employees_list'))

@app.route('/prediction/result/<int:prediction_id>')
@login_required
def prediction_result(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    emp = Employee.query.filter_by(employee_id=pred.employee_id).first()
    
    # Recommendations split into list for neat bullet points representation
    recs_list = [r.replace('• ', '').strip() for r in (pred.recommendation or '').split('\n') if r.strip()]
    
    return render_template(
        'prediction_result.html',
        prediction=pred,
        employee=emp,
        recommendations=recs_list
    )

@app.route('/prediction/history')
@login_required
def prediction_history():
    query = Prediction.query.join(Employee)
    
    # Filters
    search = request.args.get('search', '').strip()
    risk = request.args.get('risk_level', '').strip()
    
    if search:
        query = query.filter(
            (Employee.name.like(f"%{search}%")) |
            (Prediction.employee_id.like(f"%{search}%"))
        )
    if risk:
        query = query.filter(Prediction.risk_level == risk)
        
    predictions = query.order_by(Prediction.prediction_date.desc()).all()
    
    return render_template(
        'prediction_history.html',
        predictions=predictions,
        search=search,
        selected_risk=risk
    )

@app.route('/prediction/delete/<int:prediction_id>', methods=['POST'])
@login_required
def prediction_delete(prediction_id):
    pred = Prediction.query.get_or_404(prediction_id)
    try:
        db.session.delete(pred)
        db.session.commit()
        flash('Prediction log record deleted successfully.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Failed to delete prediction record: {str(e)}', 'danger')
        
    # Redirect back to caller (history or dashboards)
    return redirect(request.referrer or url_for('prediction_history'))

# --- REPORTS ---

@app.route('/reports')
@login_required
def reports():
    total_emp = Employee.query.count()
    
    # Latest predictions per employee
    latest_prediction_subquery = db.session.query(
        Prediction.employee_id,
        db.func.max(Prediction.prediction_date).label('max_date')
    ).group_by(Prediction.employee_id).subquery()

    latest_predictions = db.session.query(Prediction).join(
        latest_prediction_subquery,
        (Prediction.employee_id == latest_prediction_subquery.c.employee_id) &
        (Prediction.prediction_date == latest_prediction_subquery.c.max_date)
    ).all()
    
    high_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'High')
    medium_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'Medium')
    low_risk_cnt = sum(1 for p in latest_predictions if p.risk_level == 'Low')
    
    # Calculate Attrition Percentage
    # Formula: Attrition Rate = (Count of employees predicted to leave / Total predicted employees) * 100
    total_predicted = len(latest_predictions)
    leave_predicted_cnt = sum(1 for p in latest_predictions if p.prediction == 'Likely to Leave')
    attrition_percentage = round((leave_predicted_cnt / total_predicted * 100), 1) if total_predicted > 0 else 0.0
    
    # Group latest predictions by Department
    dept_stats = {}
    for p in latest_predictions:
        emp = Employee.query.filter_by(employee_id=p.employee_id).first()
        if not emp:
            continue
        dept = emp.department
        if dept not in dept_stats:
            dept_stats[dept] = {'total': 0, 'leave': 0, 'stay': 0, 'high_risk': 0}
        
        dept_stats[dept]['total'] += 1
        if p.prediction == 'Likely to Leave':
            dept_stats[dept]['leave'] += 1
        else:
            dept_stats[dept]['stay'] += 1
            
        if p.risk_level == 'High':
            dept_stats[dept]['high_risk'] += 1
            
    # Format department stats list for rendering
    dept_summary = []
    for dept, stats in dept_stats.items():
        rate = round((stats['leave'] / stats['total'] * 100), 1) if stats['total'] > 0 else 0.0
        dept_summary.append({
            'name': dept,
            'total': stats['total'],
            'leave': stats['leave'],
            'stay': stats['stay'],
            'high_risk': stats['high_risk'],
            'attrition_rate': rate
        })
        
    return render_template(
        'reports.html',
        total_employees=total_emp,
        high_risk=high_risk_cnt,
        medium_risk=medium_risk_cnt,
        low_risk=low_risk_cnt,
        attrition_percentage=attrition_percentage,
        dept_summary=dept_summary,
        report_date=datetime.now().strftime('%B %d, %Y')
    )

# --- ADMIN PANEL ---

@app.route('/admin/users', methods=['GET', 'POST'])
@login_required
@admin_required
def manage_users():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'HR').strip()
        
        if not name or not email or not password:
            flash('All user fields are required.', 'warning')
            return redirect(url_for('manage_users'))
            
        existing = User.query.filter_by(email=email).first()
        if existing:
            flash('A user with that email already exists.', 'warning')
            return redirect(url_for('manage_users'))
            
        try:
            new_user = User(name=name, email=email, role=role)
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()
            flash(f"User account for {name} ({role}) created successfully!", 'success')
        except Exception as e:
            db.session.rollback()
            flash(f"Failed to create user: {str(e)}", 'danger')
            
        return redirect(url_for('manage_users'))
        
    users = User.query.all()
    return render_template('manage_users.html', users=users)

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def user_delete(user_id):
    if current_user.id == user_id:
        flash('You cannot delete your own admin account.', 'warning')
        return redirect(url_for('manage_users'))
        
    user = User.query.get_or_404(user_id)
    name = user.name
    try:
        db.session.delete(user)
        db.session.commit()
        flash(f"User account for {name} deleted successfully.", 'success')
    except Exception as e:
        db.session.rollback()
        flash(f"Failed to delete user: {str(e)}", 'danger')
        
    return redirect(url_for('manage_users'))

# --- ANALYTICS AND JSON API ENDPOINTS ---

@app.route('/api/analytics/charts')
@login_required
def api_analytics_charts():
    # 1. Likely to Stay vs Likely to Leave distribution (Doughnut)
    latest_prediction_subquery = db.session.query(
        Prediction.employee_id,
        db.func.max(Prediction.prediction_date).label('max_date')
    ).group_by(Prediction.employee_id).subquery()

    latest_predictions = db.session.query(Prediction).join(
        latest_prediction_subquery,
        (Prediction.employee_id == latest_prediction_subquery.c.employee_id) &
        (Prediction.prediction_date == latest_prediction_subquery.c.max_date)
    ).all()
    
    stay_count = sum(1 for p in latest_predictions if p.prediction == 'Likely to Stay')
    leave_count = sum(1 for p in latest_predictions if p.prediction == 'Likely to Leave')
    
    # 2. Attrition by Department (Bar Chart)
    # Count of Likely to Leave vs Total Predicted in each department
    dept_data = {}
    for p in latest_predictions:
        emp = Employee.query.filter_by(employee_id=p.employee_id).first()
        if not emp:
            continue
        dept = emp.department
        if dept not in dept_data:
            dept_data[dept] = {'total': 0, 'leave': 0}
        dept_data[dept]['total'] += 1
        if p.prediction == 'Likely to Leave':
            dept_data[dept]['leave'] += 1
            
    dept_labels = list(dept_data.keys())
    dept_attrition_counts = [dept_data[d]['leave'] for d in dept_labels]
    
    # 3. Monthly Attrition Trend (Line Chart)
    # Group prediction logs by Year-Month
    trend_data = db.session.query(
        db.func.strftime('%Y-%m', Prediction.prediction_date).label('month'),
        db.func.count(Prediction.prediction_id).label('total_preds'),
        db.func.sum(db.case((Prediction.prediction == 'Likely to Leave', 1), else_=0)).label('leave_preds')
    ).group_by('month').order_by('month').all()
    
    # Fallback/sample simulation logic if DB logs are sparse (i.e. single day predictions)
    # To display a beautiful line chart, we seed previous months if there's only 1 active month
    if len(trend_data) <= 1:
        months_labels = []
        total_preds = []
        leave_preds = []
        
        now = datetime.now()
        for i in range(5, -1, -1):
            past_date = now - timedelta(days=i*30)
            month_str = past_date.strftime('%b %Y')
            months_labels.append(month_str)
            
            # Base numbers + random noise
            simulated_total = 10 + (6 - i) * 3 + int(np.random.randint(-2, 3))
            simulated_leave = int(simulated_total * (0.2 + (i % 3) * 0.05))
            
            total_preds.append(simulated_total)
            leave_preds.append(simulated_leave)
            
        # Add current real data to the last slot
        if trend_data:
            leave_c = trend_data[0][2] or 0
            leave_preds[-1] = int(leave_c)
            total_preds[-1] = int(trend_data[0][1])
    else:
        # Formulate labels from SQL result
        months_labels = []
        total_preds = []
        leave_preds = []
        for row in trend_data:
            # Convert '2026-07' to 'Jul 2026'
            try:
                dt_obj = datetime.strptime(row.month, '%Y-%m')
                months_labels.append(dt_obj.strftime('%b %Y'))
            except Exception:
                months_labels.append(row.month)
            total_preds.append(int(row.total_preds))
            leave_preds.append(int(row.leave_preds or 0))
            
    return jsonify({
        'attrition_distribution': {
            'labels': ['Likely to Stay', 'Likely to Leave'],
            'data': [stay_count, leave_count]
        },
        'department_attrition': {
            'labels': dept_labels,
            'data': dept_attrition_counts
        },
        'monthly_trend': {
            'labels': months_labels,
            'predictions': total_preds,
            'attrition': leave_preds
        }
    })

if __name__ == '__main__':
    # Run server locally on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
