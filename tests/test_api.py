#Import TestClient so we can simulated HTTP requests
from fastapi.testclient import TestClient

# Import application object defined in main.py
from main import app

# Reusable test client for all tests functions
client = TestClient(app)

def test_health_endpoint():
  """ 
  Verify that the application is running and can respond to a health check request.
  """

  response = client.get("/health")
  assert response.status_code == 200
  assert response.json() == {"status": "API is ready to server..."}


def test_valid_iris_prediction():
  """
  Verify that we are getting valid pridictions
  """
  payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }
  # send a simulated POST request with JSON input
  response = client.post("/predict", json=payload)
  # End point should  accept the valid input 
  assert response.status_code == 200
  result = response.json()
  # Verify that the prediction is valid 
  assert r"prediction" in result
  assert "class_name" in result

  assert result["class_name"] in ["setosa", "versicolor", "virginica"]
  assert result["prediction"] in [0, 1, 2]