import pytest
from app import app
import json

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'ok'

def test_predict_endpoint(client):
    """Test the prediction endpoint with sample data"""
    test_data = {
        'age': 30,
        'jobRole': 'developer',
        'monthlyIncome': 5000,
        'yearsAtCompany': 3,
        'overTime': False
    }
    
    response = client.post(
        '/api/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    # Check response structure
    assert 'risk' in data
    assert 'drivers' in data
    assert 'raw_score' in data
    
    # Check risk score is between 0 and 1
    assert 0 <= data['risk'] <= 1

def test_predict_missing_fields(client):
    """Test prediction with missing fields"""
    test_data = {
        'age': 30,
        # Missing other required fields
    }
    
    response = client.post(
        '/api/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_invalid_data(client):
    """Test prediction with invalid data types"""
    test_data = {
        'age': 'not a number',
        'jobRole': 'developer',
        'monthlyIncome': 'five thousand',
        'yearsAtCompany': 'three',
        'overTime': 'yes'
    }
    
    response = client.post(
        '/api/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    # Should return 400 for invalid data types
    assert response.status_code == 400

if __name__ == '__main__':
    pytest.main()
