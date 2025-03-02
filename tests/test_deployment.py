import requests

BASE_URL = "http://127.0.0.1:5000"

# Test health endpoint
health_response = requests.get(f"{BASE_URL}/health")
print("Health Check Response:", health_response.json())

# Test predict endpoint with valid input
test_data = {"features": [5.1, 3.5, 1.4, 0.2]}  # Example input
predict_response = requests.post(f"{BASE_URL}/predict", json=test_data)
print("Prediction Response:", predict_response.json())

# Test predict endpoint with invalid input
invalid_data = {"wrong_key": [5.1, 3.5, 1.4, 0.2]}
invalid_response = requests.post(f"{BASE_URL}/predict", json=invalid_data)
print("Invalid Request Response:", invalid_response.json())
