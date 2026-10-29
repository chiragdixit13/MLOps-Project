from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_check_endpoint():
    response = client.get("/check")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Diabetes Prediction API is working"
    }


def test_predict_endpoint():
    payload = {
        "age": 0.0380759,
        "sex": 0.0506801,
        "bmi": 0.0616962,
        "bp": 0.0218724,
        "s1": -0.0442235,
        "s2": -0.0348208,
        "s3": -0.0434008,
        "s4": -0.0025923,
        "s5": 0.0199075,
        "s6": -0.0176461
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))