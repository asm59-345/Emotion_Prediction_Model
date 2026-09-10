import pytest
import io
from starlette.testclient import TestClient
from src.main_prod import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_paragraph_flow_endpoint(client):
    story = (
        "I was walking in the dark and felt so terrified. "
        "Suddenly, all my friends jumped out with gifts and fireworks! "
        "I was completely amazed and overjoyed by the wonderful surprise."
    )
    res = client.post("/api/v1/predict/flow", json={"text": story})
    assert res.status_code == 200
    data = res.json()
    assert data["total_sentences"] >= 2
    assert "narrative_arc" in data
    assert len(data["narrative_arc"]) >= 2
    assert "dominant_emotion" in data
    assert "total_latency_ms" in data

def test_upload_csv_endpoint(client):
    csv_content = b"I feel so happy today\nI feel furious about this\nI feel so loved and cherished\n"
    files = {"file": ("test_sentences.csv", io.BytesIO(csv_content), "text/csv")}
    res = client.post("/api/v1/predict/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows_processed"] == 3
    assert data["dominant_emotion"] in ["joy", "anger", "love"]
    assert "emotion_distribution" in data
    assert len(data["sample_predictions"]) == 3

def test_upload_txt_endpoint(client):
    txt_content = b"I am terrified of heights.\nI am so proud and happy.\n"
    files = {"file": ("test_reviews.txt", io.BytesIO(txt_content), "text/plain")}
    res = client.post("/api/v1/predict/upload", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["total_rows_processed"] == 2
    assert "fear" in data["emotion_distribution"]
    assert "joy" in data["emotion_distribution"]
