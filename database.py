import os
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model, UserMixin):
    """
    User model for HR Managers, Admins, and Employees.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='HR') # 'HR', 'Admin', or 'Employee'
    employee_id = db.Column(db.String(50), db.ForeignKey('employees.employee_id', ondelete='SET NULL'), nullable=True)
    
    # Relationship to Employee
    employee_profile = db.relationship('Employee', backref='user_account', uselist=False)
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password, password)
        
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'employee_id': self.employee_id
        }

class Employee(db.Model):
    """
    Employee records containing descriptive demographic, role, and engagement features.
    """
    __tablename__ = 'employees'
    
    employee_id = db.Column(db.String(50), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    job_role = db.Column(db.String(100), nullable=False)
    salary = db.Column(db.Float, nullable=False)
    monthly_income = db.Column(db.Float, nullable=False)
    years_at_company = db.Column(db.Integer, nullable=False)
    education = db.Column(db.String(100), nullable=False)
    marital_status = db.Column(db.String(50), nullable=False)
    work_life_balance = db.Column(db.Integer, nullable=False) # 1-4 scale
    job_satisfaction = db.Column(db.Integer, nullable=False) # 1-4 scale
    performance_rating = db.Column(db.Integer, nullable=False) # 1-4 scale
    overtime = db.Column(db.String(5), nullable=False) # 'Yes' or 'No'
    distance_from_home = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    # Cascade delete predictions if employee is deleted
    predictions = db.relationship('Prediction', backref='employee', cascade='all, delete-orphan', lazy=True)
    
    def to_dict(self):
        return {
            'employee_id': self.employee_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'department': self.department,
            'job_role': self.job_role,
            'salary': self.salary,
            'monthly_income': self.monthly_income,
            'years_at_company': self.years_at_company,
            'education': self.education,
            'marital_status': self.marital_status,
            'work_life_balance': self.work_life_balance,
            'job_satisfaction': self.job_satisfaction,
            'performance_rating': self.performance_rating,
            'overtime': self.overtime,
            'distance_from_home': self.distance_from_home,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S') if self.created_at else None
        }

class Prediction(db.Model):
    """
    Log of predictive inferences, probabilities, risk categories, and dynamic recommendations.
    """
    __tablename__ = 'predictions'
    
    prediction_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    employee_id = db.Column(db.String(50), db.ForeignKey('employees.employee_id', ondelete='CASCADE'), nullable=False)
    prediction = db.Column(db.String(50), nullable=False) # 'Likely to Stay' or 'Likely to Leave'
    probability = db.Column(db.Float, nullable=False) # leave probability (0 to 100)
    risk_level = db.Column(db.String(20), nullable=False) # 'Low', 'Medium', 'High'
    recommendation = db.Column(db.Text, nullable=True)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'prediction_id': self.prediction_id,
            'employee_id': self.employee_id,
            'prediction': self.prediction,
            'probability': self.probability,
            'risk_level': self.risk_level,
            'recommendation': self.recommendation,
            'prediction_date': self.prediction_date.strftime('%Y-%m-%d %H:%M:%S') if self.prediction_date else None,
            'employee_name': self.employee.name if self.employee else 'Unknown'
        }

def init_database(app):
    """
    Configures and initializes the database with the Flask application.
    """
    # Fetch connection string from environment if defined (for MySQL/Postgres), fallback to SQLite
    db_uri = os.environ.get('DATABASE_URL')
    
    if not db_uri:
        # Check standard config file or fallback to default SQLite
        db_path = os.path.join(app.root_path, 'attrition.db')
        db_uri = f'sqlite:///{db_path}'
        
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
