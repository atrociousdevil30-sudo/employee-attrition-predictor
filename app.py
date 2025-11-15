from flask import Flask, request, jsonify, send_from_directory
import os
import json
from model import AttritionPredictor
import joblib
from datetime import datetime
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder='static')

# In-memory log of high-risk predictions for internal verification
high_risk_predictions = []

# Initialize the predictor
predictor = AttritionPredictor()

# Try to load pre-trained model
try:
    model_loaded = predictor.load_model(
        model_path='model/attrition_model.joblib',
        preprocessor_path='model/preprocessor.joblib'
    )
    if not model_loaded:
        print("Warning: Could not load pre-trained model. Training a new one...")
        from model import train_and_save_model
        train_and_save_model()
        model_loaded = predictor.load_model(
            model_path='model/attrition_model.joblib',
            preprocessor_path='model/preprocessor.joblib'
        )
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model_loaded = False

def prepare_input_data(form_data):
    """Prepare input data for the model prediction"""
    # Map form field names to model feature names
    input_data = {
        'Age': int(form_data.get('age', 30)),
        'BusinessTravel': 'Travel_Rarely',  # Default value
        'DailyRate': 800,  # Default value
        'Department': 'Research & Development',  # Default value
        'DistanceFromHome': 10,  # Default value
        'Education': 3,  # Default value (1-5)
        'EducationField': 'Life Sciences',  # Default value
        'EnvironmentSatisfaction': 3,  # Default value (1-4)
        'Gender': 'Male' if form_data.get('gender', 'male') == 'male' else 'Female',
        'HourlyRate': 65,  # Default value
        'JobInvolvement': 3,  # Default value (1-4)
        'JobLevel': 2,  # Default value (1-5)
        'JobRole': form_data.get('jobRole', 'Research Scientist'),
        'JobSatisfaction': 3,  # Default value (1-4)
        'MonthlyIncome': int(form_data.get('monthlyIncome', 5000)),
        'MonthlyRate': 15000,  # Default value
        'NumCompaniesWorked': 2,  # Default value
        'OverTime': 'Yes' if str(form_data.get('overTime')).lower() in ['true', '1'] else 'No',
        'PercentSalaryHike': 14,  # Default value
        'PerformanceRating': 3,  # Default value (1-4)
        'StockOptionLevel': 1,  # Default value (0-3)
        'TotalWorkingYears': int(form_data.get('yearsAtCompany', 5)) + 5,  # Estimate
        'TrainingTimesLastYear': 2,  # Default value
        'WorkLifeBalance': 3,  # Default value (1-4)
        'YearsAtCompany': int(form_data.get('yearsAtCompany', 3)),
        'YearsInCurrentRole': min(3, int(form_data.get('yearsAtCompany', 3))),  # Estimate
        'YearsSinceLastPromotion': max(0, int(form_data.get('yearsAtCompany', 3)) - 2),  # Estimate
        'YearsWithCurrManager': min(2, int(form_data.get('yearsAtCompany', 3)))  # Estimate
    }
    
    return input_data

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# Serve index.html for the root URL
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/train', methods=['POST'])
def train():
    """Endpoint to retrain the model"""
    try:
        from model import train_and_save_model
        model, accuracy = train_and_save_model()
        
        if model is not None:
            # Reload the model
            global predictor, model_loaded
            model_loaded = predictor.load_model(
                model_path='model/attrition_model.joblib',
                preprocessor_path='model/preprocessor.joblib'
            )
            
            return jsonify({
                'status': 'success',
                'accuracy': accuracy,
                'message': f'Model retrained with accuracy: {accuracy:.4f}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to train the model'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error training model: {str(e)}'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "ok"})

@app.route('/api/upload_dataset', methods=['POST'])
def upload_dataset():
    """Upload IBM attrition dataset CSV and save it to local data directory."""
    if 'dataset' not in request.files:
        return jsonify({"error": "No dataset file part in request"}), 400

    file = request.files['dataset']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read a small portion into pandas to validate structure
        df = pd.read_csv(file)

        required_columns = {"Age", "MonthlyIncome", "JobRole", "YearsAtCompany", "Attrition"}
        if not required_columns.issubset(df.columns):
            return jsonify({
                "error": "Uploaded CSV does not match IBM attrition structure",
                "missing_columns": list(required_columns - set(df.columns))
            }), 400

        os.makedirs('data', exist_ok=True)
        save_path = os.path.join('data', 'emp_attrition.csv')
        df.to_csv(save_path, index=False)

        # Pick a sample employee row to help auto-fill the form on the frontend
        sample = df.iloc[0]
        sample_employee = {
            "Age": int(sample["Age"]),
            "Gender": str(sample.get("Gender", "Male")),
            "JobRole": str(sample["JobRole"]),
            "MonthlyIncome": int(sample["MonthlyIncome"]),
            "YearsAtCompany": int(sample["YearsAtCompany"]),
            "OverTime": str(sample.get("OverTime", "No"))
        }

        # Build a compact list of employees for selection on the frontend
        # Limit to first 200 rows to keep payload reasonable
        cols_for_frontend = [
            col for col in [
                "EmployeeNumber", "Age", "Gender", "JobRole",
                "MonthlyIncome", "YearsAtCompany", "OverTime"
            ] if col in df.columns
        ]
        employees = df[cols_for_frontend].head(200).to_dict(orient="records")

        return jsonify({
            "status": "success",
            "message": "Dataset uploaded and saved as data/emp_attrition.csv",
            "rows": int(df.shape[0]),
            "sample_employee": sample_employee,
            "employees": employees
        })

    except Exception as e:
        return jsonify({"error": f"Error processing dataset: {str(e)}"}, 500)

@app.route('/api/job_roles', methods=['GET'])
def get_job_roles():
    if not model_loaded or predictor.preprocessor is None:
        app.logger.error("Model or preprocessor not loaded")
        return jsonify({
            "error": "Prediction service is not ready",
            "details": "Model or preprocessor not loaded"
        }), 503
    
    try:
        # Default job roles as fallback
        default_job_roles = [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ]

        # Try to get job roles from the preprocessor
        job_roles = []
        
        # Check if we have transformers
        if not hasattr(predictor.preprocessor, 'transformers_'):
            app.logger.warning("No transformers found in preprocessor, using default job roles")
            return jsonify({"job_roles": default_job_roles})
            
        # Look through all transformers
        for name, transformer, features in predictor.preprocessor.transformers_:
            if not hasattr(transformer, 'named_steps'):
                continue
                
            if 'onehot' in transformer.named_steps:
                onehot = transformer.named_steps['onehot']
                if hasattr(onehot, 'get_feature_names_out'):
                    # Get all feature names from this transformer
                    feature_names = onehot.get_feature_names_out(features)
                    # Filter for JobRole features
                    job_roles = [
                        name.split('_', 1)[1] for name in feature_names 
                        if name.startswith('JobRole_')
                    ]
                    if job_roles:
                        return jsonify({"job_roles": job_roles})
        
        # If we get here, we couldn't find job roles in the preprocessor
        app.logger.warning("Could not extract job roles from preprocessor, using defaults")
        return jsonify({"job_roles": default_job_roles})
        
    except Exception as e:
        app.logger.error(f"Error getting job roles: {str(e)}", exc_info=True)
        # Return default roles even in case of error
        return jsonify({"job_roles": default_job_roles})

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model_loaded:
        app.logger.error("Prediction model not available")
        return jsonify({
            "error": "Prediction service is currently unavailable",
            "details": "Model failed to load"
        }), 503  # Service Unavailable
        
    try:
        # Check if request has JSON data
        if not request.is_json:
            return jsonify({
                "error": "Invalid request format",
                "details": "Request must be JSON"
            }), 400

        data = request.get_json()
        
        # Validate required fields
        required_fields = {
            'age': (int, "must be a number"),
            'jobRole': (str, "must be a string"),
            'monthlyIncome': ((int, float), "must be a number"),
            'yearsAtCompany': ((int, float), "must be a number"),
            'overTime': (bool, "must be a boolean")
        }
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                "error": "Missing required fields",
                "missing": missing_fields
            }), 400

        # Validate field types
        type_errors = []
        for field, (expected_type, error_msg) in required_fields.items():
            value = data[field]
            if not isinstance(value, expected_type) and not (isinstance(expected_type, tuple) and any(isinstance(value, t) for t in expected_type)):
                type_errors.append(f"{field}: {error_msg}")

        if type_errors:
            return jsonify({
                "error": "Invalid field types",
                "details": type_errors
            }), 400

        # Validate value ranges
        validation_errors = []
        if 'age' in data and (data['age'] < 18 or data['age'] > 100):
            validation_errors.append("age: must be between 18 and 100")
        if 'monthlyIncome' in data and data['monthlyIncome'] < 0:
            validation_errors.append("monthlyIncome: must be a positive number")
        if 'yearsAtCompany' in data and data['yearsAtCompany'] < 0:
            validation_errors.append("yearsAtCompany: must be a positive number")

        if validation_errors:
            return jsonify({
                "error": "Invalid field values",
                "details": validation_errors
            }), 400
        
        # Prepare input data for the model
        input_data = prepare_input_data(data)
        
        # Make prediction
        prediction = predictor.predict(input_data)
        
        # Format the response
        risk_score = prediction['probability']
        
        # Determine risk level (slightly stricter thresholds so more borderline cases are Medium/High)
        if risk_score < 0.4:
            risk_level = "Low"
        elif risk_score < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # Business rule override: if employee works overtime and model sees at least some risk,
        # do not leave them tagged as Low risk.
        overtime_flag = input_data.get('OverTime')
        if overtime_flag == 'Yes' and risk_level == 'Low' and risk_score >= 0.25:
            risk_level = 'Medium'

        # Build personalized retention suggestions based on employee's specific situation
        retention_suggestions = []
        
        # Get employee data
        yrs = input_data.get('YearsAtCompany')
        age_val = input_data.get('Age')
        income_val = input_data.get('MonthlyIncome')
        role_val = str(input_data.get('JobRole') or '')
        overtime = str(input_data.get('OverTime', 'No'))
        
        # Base suggestions based on risk level
        if risk_level == "High":
            retention_suggestions.append("High Priority: Schedule an immediate 1:1 meeting to address concerns and develop a retention plan.")
        elif risk_level == "Medium":
            retention_suggestions.append("Schedule a career development discussion within the next 2 weeks.")
        
        # Tenure-based suggestions
        if yrs is not None:
            if yrs < 1:
                retention_suggestions.append("New Hire Support: Implement 30/60/90-day check-ins and assign an onboarding buddy.")
            elif yrs < 3:
                retention_suggestions.append("Career Development: Discuss career path and development opportunities during growth phase.")
            elif yrs >= 5:
                retention_suggestions.append("Career Growth: Explore lateral moves or stretch assignments to maintain engagement.")
        
        # Overtime and workload
        if overtime == 'Yes':
            retention_suggestions.append("Workload Management: Review and rebalance workload to reduce overtime and prevent burnout.")
        
        # Compensation considerations
        if income_val is not None:
            if risk_level in ['High', 'Medium'] and (income_val < 4000 or 'Sales' in role_val):
                retention_suggestions.append("Compensation Review: Benchmark salary against market rates and consider adjustments.")
        
        # Age and career stage
        if age_val is not None:
            if age_val < 30:
                retention_suggestions.append("Professional Development: Create a clear skill development plan with milestones.")
            elif age_val >= 45:
                retention_suggestions.append("Experience Utilization: Leverage expertise through mentoring or special projects.")
        
        # Role-specific suggestions
        if 'Sales' in role_val:
            retention_suggestions.append("Sales Performance: Review targets, provide enablement tools, and ensure competitive commission structure.")
        elif 'Research' in role_val:
            retention_suggestions.append("Research Support: Support conference attendance and publication opportunities.")
        elif 'Human Resources' in role_val:
            retention_suggestions.append("HR Development: Offer leadership training and cross-functional project opportunities.")
        
        # Work-life balance
        if overtime == 'Yes' or risk_level == 'High':
            retention_suggestions.append("Work-Life Balance: Review workload and consider flexible work arrangements if possible.")
        
        # Add general best practices if we don't have many specific suggestions
        if len(retention_suggestions) < 3:
            retention_suggestions.extend([
                "Career Development: Schedule regular career development conversations",
                "Performance Management: Set clear performance and growth objectives",
                "Recognition: Acknowledge achievements and contributions regularly"
            ])
        
        # Ensure we don't have too many suggestions
        retention_suggestions = retention_suggestions[:8]  # Limit to 8 most relevant suggestions
        
        # Build human-readable factors for the analysis based on inputs and risk level
        detailed_factors = []

        age = input_data.get('Age')
        years_at_company = input_data.get('YearsAtCompany')
        monthly_income = input_data.get('MonthlyIncome')
        overtime_flag = input_data.get('OverTime')
        job_role = input_data.get('JobRole', '')
        
        # Base demographic information
        if age is not None and years_at_company is not None:
            year_text = "year" if years_at_company == 1 else "years"
            detailed_factors.append(f"{age} years old and working for {years_at_company} {year_text}.")

        # Overtime analysis
        if overtime_flag == 'Yes':
            detailed_factors.append("Employee is working overtime regularly, which may impact work-life balance.")
        else:
            detailed_factors.append("Employee maintains standard work hours, supporting work-life balance.")

        # Compensation analysis
        if monthly_income is not None:
            if risk_level == 'High':
                detailed_factors.append(f"Current compensation (${monthly_income:,.0f}) may not meet market expectations for this role.")
            elif risk_level == 'Medium':
                detailed_factors.append(f"Current compensation (${monthly_income:,.0f}) should be benchmarked against industry standards.")
            else:
                detailed_factors.append(f"Current compensation (${monthly_income:,.0f}) appears appropriate for this role.")

        # Tenure analysis
        if years_at_company is not None:
            try:
                # Ensure years_at_company is a number
                years = float(years_at_company)
                if years < 1:
                    detailed_factors.append("New employee in critical first year, requiring focused onboarding and support.")
                elif 1 <= years < 2:
                    detailed_factors.append("Employee has completed first year; focus on career development and engagement to prevent second-year attrition.")
                elif 2 <= years < 3:
                    detailed_factors.append("Employee is approaching the typical turnover point (2-3 years) where career growth expectations heighten.")
                elif 3 <= years < 7:
                    detailed_factors.append("Established employee with company tenure, indicating good cultural fit.")
                else:
                    detailed_factors.append("Long-tenured employee who may benefit from new challenges to maintain engagement.")
            except (ValueError, TypeError) as e:
                app.logger.error(f"Error processing tenure analysis: {str(e)}")
                detailed_factors.append("Tenure analysis not available due to data format issues.")

        # Age-based considerations
        if age is not None:
            if age < 30:
                detailed_factors.append("Younger employee likely seeking career development and skill growth opportunities.")
            elif age >= 45:
                detailed_factors.append("Experienced employee who values stability, recognition, and work-life balance.")

        # Risk level summary
        if risk_level == 'High':
            detailed_factors.append("High Risk: Multiple factors indicate elevated attrition risk requiring immediate attention.")
        elif risk_level == 'Medium':
            detailed_factors.append("Moderate Risk: Some risk factors present that could impact retention if unaddressed.")
        else:
            detailed_factors.append("Low Risk: Current indicators suggest good employee engagement and retention potential.")
            
        # Add any specific recommendations
        if risk_level in ['High', 'Medium']:
            detailed_factors.append("Recommendation: Proactive measures recommended to address potential risk factors.")

        # Keep an internal log of high-risk employees for verification (not exposed as a feature)
        if risk_level == "High":
            high_risk_entry = {
                "age": input_data.get("Age"),
                "jobRole": input_data.get("JobRole"),
                "monthlyIncome": input_data.get("MonthlyIncome"),
                "yearsAtCompany": input_data.get("YearsAtCompany"),
                "risk": float(risk_score),
                "timestamp": datetime.now().isoformat()
            }
            high_risk_predictions.append(high_risk_entry)
            app.logger.info(f"High-risk prediction: {json.dumps(high_risk_entry)}")
        
        return jsonify({
            'risk': round(risk_score, 3),
            'risk_level': risk_level,
            'factors': detailed_factors,
            'retention_suggestions': retention_suggestions,
            'raw_prediction': float(prediction['prediction']),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred while processing your request',
            'details': str(e)
        }), 500

@app.route('/api/find_similar', methods=['POST'])
def find_similar():
    try:
        data = request.get_json()
        if not os.path.exists('data/emp_attrition.csv'):
            return jsonify({"error": "Dataset not available"}), 400
        input_data = prepare_input_data(data)
        df = pd.read_csv('data/emp_attrition.csv')
        numeric_cols = [c for c in ['Age', 'YearsAtCompany', 'MonthlyIncome'] if c in df.columns]
        cat_cols = [c for c in ['JobRole', 'OverTime'] if c in df.columns]
        if not numeric_cols:
            return jsonify({"error": "Required numeric columns not found in dataset"}), 400
        s = df[numeric_cols].std().replace(0, 1)
        eps = 1e-6
        target_num = pd.Series({c: float(input_data.get(c, 0)) for c in numeric_cols})
        num_diff = (df[numeric_cols] - target_num)
        num_dist = (num_diff.pow(2)).div((s.pow(2) + eps), axis=1).sum(axis=1)
        cat_dist = pd.Series(0.0, index=df.index)
        if 'JobRole' in cat_cols:
            cat_dist += (df['JobRole'].astype(str) != str(input_data.get('JobRole'))).astype(float)
        if 'OverTime' in cat_cols:
            cat_dist += 0.5 * (df['OverTime'].astype(str) != str(input_data.get('OverTime'))).astype(float)
        total_dist = num_dist + cat_dist
        k = int(data.get('k', 5))
        top_idx = total_dist.nsmallest(k).index
        results = []
        for idx in top_idx:
            row = df.loc[idx]
            item = {
                "index": int(idx),
                "Age": int(row['Age']) if 'Age' in df.columns else None,
                "YearsAtCompany": int(row['YearsAtCompany']) if 'YearsAtCompany' in df.columns else None,
                "MonthlyIncome": int(row['MonthlyIncome']) if 'MonthlyIncome' in df.columns else None,
                "JobRole": str(row['JobRole']) if 'JobRole' in df.columns else None,
                "OverTime": str(row['OverTime']) if 'OverTime' in df.columns else None,
                "Attrition": str(row['Attrition']) if 'Attrition' in df.columns else None,
                "distance": float(total_dist.loc[idx]),
                "similarity": round(1.0 / (1.0 + float(total_dist.loc[idx])), 3)
            }
            if 'EmployeeNumber' in df.columns:
                item['EmployeeNumber'] = int(row['EmployeeNumber'])
            results.append(item)
        return jsonify({"similar": results})
    except Exception as e:
        return jsonify({"error": f"Error finding similar employees: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
