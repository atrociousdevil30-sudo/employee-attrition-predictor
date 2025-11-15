import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import os

class AttritionPredictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        # Define available features from HRDataset_v14.csv
        self.feature_columns = [
            'Age', 'Gender', 'MaritalStatus', 'Department', 'Position',
            'Salary', 'EmpSatisfaction', 'EngagementSurvey', 'SpecialProjectsCount',
            'DaysLateLast30', 'Absences', 'YearsAtCompany', 'PerformanceScore'
        ]
        
        # Categorical features (will be one-hot encoded)
        self.categorical_features = [
            'Gender', 'MaritalStatus', 'Department', 'Position', 'PerformanceScore'
        ]
        
        # Numerical features (will be scaled)
        self.numerical_features = [
            'Age', 'Salary', 'EmpSatisfaction', 'EngagementSurvey',
            'SpecialProjectsCount', 'DaysLateLast30', 'Absences', 'YearsAtCompany'
        ]
        
        # Ordinal features (categorical with inherent order)
        self.ordinal_features = {
            'EmpSatisfaction': [1, 2, 3, 4, 5],
            'EngagementSurvey': [1, 2, 3, 4, 5],
            'PerformanceScore': [1, 2, 3, 4]  # Assuming 1=Needs Improvement, 2=Meets, 3=Exceeds, 4=Outstanding
        }

    def load_data(self, file_path):
        """Load and preprocess the HR Attrition dataset from local file"""
        try:
            # Load the dataset
            df = pd.read_csv(file_path)
            
            # Map the column names to match our expected format
            column_mapping = {
                'EmpID': 'EmployeeID',
                'MaritalStatusID': 'MaritalStatus',
                'MaritalDesc': 'MaritalStatus',
                'GenderID': 'Gender',
                'Sex': 'Gender',
                'EmpStatusID': 'EmployeeStatus',
                'DeptID': 'Department',
                'PerfScoreID': 'PerformanceScore',
                'FromDiversityJobFairID': 'FromDiversityJobFair',
                'Termd': 'Attrition',
                'PositionID': 'PositionID',
                'Position': 'Position',
                'Salary': 'Salary',
                'EmpSatisfaction': 'EmpSatisfaction',
                'EngagementSurvey': 'EngagementSurvey',
                'SpecialProjectsCount': 'SpecialProjectsCount',
                'DaysLateLast30': 'DaysLateLast30',
                'Absences': 'Absences',
                'DOB': 'DOB',
                'DateofHire': 'DateofHire',
                'DateofTermination': 'TerminationDate',
                'TermReason': 'TerminationReason',
                'EmploymentStatus': 'EmployeeStatus',
                'RecruitmentSource': 'RecruitmentSource'
            }
            
            # Rename columns to match expected format
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Calculate Age from DOB (assuming current year is 2023 for this dataset)
            if 'DOB' in df.columns:
                df['Age'] = 2023 - pd.to_datetime(df['DOB'], errors='coerce').dt.year
                df['Age'] = df['Age'].fillna(df['Age'].median())
            else:
                # If DOB is not available, use a default age of 35
                df['Age'] = 35
            
            # Calculate YearsAtCompany from DateofHire
            if 'DateofHire' in df.columns:
                df['DateofHire'] = pd.to_datetime(df['DateofHire'], errors='coerce')
                df['YearsAtCompany'] = 2023 - df['DateofHire'].dt.year
                df['YearsAtCompany'] = df['YearsAtCompany'].fillna(df['YearsAtCompany'].median())
            else:
                # If DateofHire is not available, use a default of 5 years
                df['YearsAtCompany'] = 5
                
            # Fill any missing values
            df['EmpSatisfaction'] = df['EmpSatisfaction'].fillna(df['EmpSatisfaction'].median())
            df['EngagementSurvey'] = df['EngagementSurvey'].fillna(df['EngagementSurvey'].median())
            df['SpecialProjectsCount'] = df['SpecialProjectsCount'].fillna(0)
            df['DaysLateLast30'] = df['DaysLateLast30'].fillna(0)
            df['Absences'] = df['Absences'].fillna(0)
            
            # Convert Attrition to binary (1 for terminated, 0 for still employed)
            if 'Attrition' in df.columns:
                df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 1 else 0)
            else:
                # If Attrition column doesn't exist, create it based on TerminationDate
                df['Attrition'] = df['TerminationDate'].apply(lambda x: 0 if pd.isna(x) else 1)
            
            # Select and return relevant columns
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Separate features and target
        X = df.drop('Attrition', axis=1)
        y = df['Attrition']
        
        # Create transformers for numerical and categorical features
        numeric_transformer = StandardScaler()
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        # Apply transformations
        X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y

    def train_model(self, X, y):
        """Train the Random Forest Classifier"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained with accuracy: {accuracy:.4f}")
        
        return self.model, accuracy

    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None or self.preprocessor is None:
            raise Exception("Model not trained. Please train the model first.")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply the same preprocessing
        try:
            processed_data = self.preprocessor.transform(input_df)
            
            # Make prediction
            probability = self.model.predict_proba(processed_data)[0][1]  # Probability of attrition (class 1)
            prediction = self.model.predict(processed_data)[0]
            
            # Get feature importances
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                
                # Get feature names after one-hot encoding
                try:
                    # For numerical features
                    feature_names = self.numerical_features.copy()
                    
                    # For one-hot encoded categorical features
                    if hasattr(self.preprocessor.named_transformers_['cat'].named_steps['onehot'], 'get_feature_names_out'):
                        cat_features = self.preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
                        feature_names.extend(cat_features)
                    
                    # Get top 3 important features
                    top_indices = importances.argsort()[-3:][::-1]
                    top_features = [feature_names[i] for i in top_indices]
                    
                    return {
                        'probability': float(probability),
                        'prediction': int(prediction),
                        'top_features': top_features
                    }
                except Exception as e:
                    print(f"Error getting feature importances: {str(e)}")
            
            return {
                'probability': float(probability),
                'prediction': int(prediction),
                'top_features': []
            }
            
        except Exception as e:
            raise Exception(f"Error in prediction: {str(e)}")

    def save_model(self, model_path='model', preprocessor_path='preprocessor.joblib'):
        """Save the trained model and preprocessor"""
        if self.model is None or self.preprocessor is None:
            raise Exception("Model or preprocessor not available. Please train the model first.")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model and preprocessor
        joblib.dump(self.model, f"{model_path}.joblib")
        joblib.dump(self.preprocessor, preprocessor_path)
        
        print(f"Model saved to {model_path}.joblib")
        print(f"Preprocessor saved to {preprocessor_path}")

    def load_model(self, model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
        """Load a trained model and preprocessor"""
        try:
            self.model = joblib.load(model_path)
            self.preprocessor = joblib.load(preprocessor_path)
            print("Model and preprocessor loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def train_and_save_model():
    """Train the model and save it to disk"""
    # Path to the local HR dataset
    dataset_path = os.path.join('data', 'HRDataset_v14.csv')
    
    # Initialize and train the model
    predictor = AttritionPredictor()
    
    # Load and preprocess data
    print("Loading data...")
    df = predictor.load_data(dataset_path)
    
    if df is not None:
        print("Preprocessing data...")
        X, y = predictor.preprocess_data(df)
        
        print("Training model...")
        model, accuracy = predictor.train_model(X, y)
        
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        # Save the trained model and preprocessor
        predictor.save_model('model/attrition_model', 'model/preprocessor.joblib')
        
        return model, accuracy
    
    return None, None

if __name__ == "__main__":
    train_and_save_model()
