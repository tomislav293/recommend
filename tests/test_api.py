import sys, os
# Always add project root (parent folder of "tests") to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.api import app


client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"text": "New fitness app for marathon training"})
    assert response.status_code == 200
    data = response.json()
    assert "assigned_cluster" in data
    assert "top_clusters" in data

