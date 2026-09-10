import pytest
from starlette.testclient import TestClient
from src.main_prod import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_predict_single_endpoint(client):
    payload = {
        "text": "I feel so grateful and joyful today!",
        "explain": True
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_emotion"] == "joy"
    assert data["confidence"] > 0.7
    assert data["sentiment"]["polarity"] == "positive"
    assert data["attributions"] is not None
    assert len(data["attributions"]) > 0

def test_predict_batch_endpoint(client):
    payload = {
        "texts": [
            "I feel extremely furious and angry!",
            "I feel so shocked and surprised by this unexpected gift"
        ],
        "explain": False
    }
    res = client.post("/api/v1/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_samples"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["predicted_emotion"] == "anger"
    assert data["predictions"][1]["predicted_emotion"] == "surprise"

def test_analytics_endpoints(client):
    # Fetch summary
    res = client.get("/api/v1/analytics/summary")
    assert res.status_code == 200
    data = res.json()
    assert "total_predictions" in data
    assert data["total_predictions"] >= 1
    
    # Fetch history
    res_hist = client.get("/api/v1/analytics/history?limit=5")
    assert res_hist.status_code == 200
    records = res_hist.json()
    assert isinstance(records, list)
    assert len(records) > 0
