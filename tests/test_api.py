from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_predict_failure():
    payload = {
        "sensor_temperature_c": 92.0,
        "vibration_mm_s": 5.5,
        "pressure_bar": 22.0,
        "rotational_speed_rpm": 1350.0,
        "operating_hours": 3500,
        "torque_nm": 68.0
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "failure_probability" in data
    assert "top_shap_drivers" in data