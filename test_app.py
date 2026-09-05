import os
import unittest
from app import app, db, User, Employee, Prediction
from ml_model import predict_attrition, load_model

class AttritionSystemTestCase(unittest.TestCase):
    
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        
        self.client = app.test_client()
        
        with app.app_context():
            db.create_all()
            # Clear existing data in dependency order to prevent constraint errors
            Prediction.query.delete()
            Employee.query.delete()
            User.query.delete()
            db.session.commit()
            
            # Seed test users
            hr_user = User(name="Test HR", email="testhr@test.com", role="HR")
            hr_user.set_password("hrpassword")
            db.session.add(hr_user)
            
            admin_user = User(name="Test Admin", email="testadmin@test.com", role="Admin")
            admin_user.set_password("adminpassword")
            db.session.add(admin_user)
            
            db.session.commit()
            
    def tearDown(self):
        with app.app_context():
            db.session.remove()
            # Clean up after test
            Prediction.query.delete()
            Employee.query.delete()
            User.query.delete()
            db.session.commit()
            
    def test_model_loading_and_prediction(self):
        """
        Verify that the serialized Random Forest model loads successfully and returns expected classifications.
        """
        # Verify model loads without throwing errors
        payload = load_model()
        self.assertIsNotNone(payload)
        self.assertEqual(payload['model_name'], 'Random Forest')
        
        # Test input matching the feature columns
        test_employee = {
            'age': 32,
            'gender': 'Male',
            'department': 'IT',
            'job_role': 'Developer',
            'salary': 60000,
            'monthly_income': 5000,
            'years_at_company': 2,
            'education': "Bachelor's Degree",
            'marital_status': 'Single',
            'work_life_balance': 2,
            'job_satisfaction': 1, # Low satisfaction
            'performance_rating': 3,
            'overtime': 'Yes', # High overtime
            'distance_from_home': 18.0
        }
        
        prediction_output = predict_attrition(test_employee)
        
        # Verify output keys
        self.assertIn('prediction', prediction_output)
        self.assertIn('leave_probability', prediction_output)
        self.assertIn('risk_level', prediction_output)
        self.assertIn('recommendations', prediction_output)
        
        # Verify probabilities range from 0 to 100
        self.assertTrue(0 <= prediction_output['leave_probability'] <= 100)
        self.assertTrue(0 <= prediction_output['stay_probability'] <= 100)
        
        # Verify risk level categorizations
        self.assertIn(prediction_output['risk_level'], ['Low', 'Medium', 'High'])

    def test_guest_redirects(self):
        """
        Check that unauthenticated guests are redirected to the sign-in page when trying to access dashboards.
        """
        response = self.client.get('/dashboard')
        self.assertEqual(response.status_code, 302)
        self.assertIn('/login', response.headers['Location'])
        
        response = self.client.get('/employees')
        self.assertEqual(response.status_code, 302)

    def test_registration_and_login(self):
        """
        Test HR manager registration and login procedures.
        """
        # Register a new HR Manager
        reg_response = self.client.post('/register', data={
            'name': 'New HR Officer',
            'email': 'newhr@company.com',
            'password': 'newpassword123'
        }, follow_redirects=True)
        self.assertEqual(reg_response.status_code, 200)
        self.assertIn(b'Registration successful', reg_response.data)
        
        # Attempt to log in via HR login endpoint
        login_response = self.client.post('/login/hr', data={
            'email': 'newhr@company.com',
            'password': 'newpassword123'
        }, follow_redirects=True)
        self.assertEqual(login_response.status_code, 200)
        self.assertIn(b'New HR Officer', login_response.data)
        
        # Test Sign Out (Logout)
        logout_response = self.client.get('/logout', follow_redirects=True)
        self.assertEqual(logout_response.status_code, 200)
        self.assertIn(b'Successfully logged out', logout_response.data)

    def test_employee_login_and_portal(self):
        """
        Test Employee login via /login/employee and access to Employee Self-Service portal.
        """
        with app.app_context():
            # Create Employee user
            emp_user = User(name="Sample Staff", email="staff@company.com", role="Employee", employee_id="EMP999")
            emp_user.set_password("staff123")
            db.session.add(emp_user)

            emp_rec = Employee(
                employee_id="EMP999", name="Sample Staff", age=30, gender="Female",
                department="Marketing", job_role="Manager", salary=90000, monthly_income=7500,
                years_at_company=5, education="Master's Degree", marital_status="Single",
                work_life_balance=3, job_satisfaction=4, performance_rating=3, overtime="No", distance_from_home=5.0
            )
            db.session.add(emp_rec)
            db.session.commit()

        # Login via /login/employee using Employee ID
        login_resp = self.client.post('/login/employee', data={
            'identifier': 'EMP999',
            'password': 'staff123'
        }, follow_redirects=True)
        self.assertEqual(login_resp.status_code, 200)
        self.assertIn(b'Sample Staff', login_resp.data)
        self.assertIn(b'EMPLOYEE SELF-SERVICE', login_resp.data)

        # Ensure Employee cannot access HR-only list
        hr_only_resp = self.client.get('/employees', follow_redirects=True)
        self.assertIn(b'HR Manager privileges required', hr_only_resp.data)
        
    def test_employee_crud_endpoints(self):
        """
        Test adding, editing, and deleting employee records under authenticated sessions.
        """
        # Log in as HR Manager
        self.client.post('/login', data={
            'email': 'testhr@test.com',
            'password': 'hrpassword'
        })
        
        # Add Employee
        add_response = self.client.post('/employee/add', data={
            'employee_id': 'EMP2001',
            'name': 'Ramesh Kumar',
            'age': '28',
            'gender': 'Male',
            'department': 'IT',
            'job_role': 'Developer',
            'salary': '72000',
            'monthly_income': '6000',
            'years_at_company': '4',
            'education': "Bachelor's Degree",
            'marital_status': 'Married',
            'work_life_balance': '3',
            'job_satisfaction': '4',
            'performance_rating': '3',
            'overtime': 'No',
            'distance_from_home': '6.0'
        }, follow_redirects=True)
        
        self.assertEqual(add_response.status_code, 200)
        self.assertIn(b'Ramesh Kumar', add_response.data)
        
        # Edit Employee
        edit_response = self.client.post('/employee/edit/EMP2001', data={
            'name': 'Ramesh Kumar (Updated)',
            'age': '29',
            'gender': 'Male',
            'department': 'IT',
            'job_role': 'Developer',
            'salary': '84000',
            'monthly_income': '7000',
            'years_at_company': '4',
            'education': "Bachelor's Degree",
            'marital_status': 'Married',
            'work_life_balance': '3',
            'job_satisfaction': '3',
            'performance_rating': '3',
            'overtime': 'Yes', # Changed
            'distance_from_home': '6.0'
        }, follow_redirects=True)
        
        self.assertEqual(edit_response.status_code, 200)
        self.assertIn(b'Ramesh Kumar (Updated)', edit_response.data)
        
        # Delete Employee
        delete_response = self.client.post('/employee/delete/EMP2001', follow_redirects=True)
        self.assertEqual(delete_response.status_code, 200)
        self.assertIn(b'deleted successfully', delete_response.data)
        
        # Verify it is no longer present in the database
        with app.app_context():
            emp_in_db = db.session.get(Employee, 'EMP2001')
            self.assertIsNone(emp_in_db)

if __name__ == '__main__':
    unittest.main()
