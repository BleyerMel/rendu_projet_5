from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_prediction():
    data = {
        'Title': ['help ! my function not working'],
        'Body': ['i need help with my python function']
    }

    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
