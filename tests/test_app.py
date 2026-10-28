"""API checks for the Flask deployment."""
import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="module")
def client():
    from app.app import app
    app.config["TESTING"] = True
    return app.test_client()


def _row():
    return pd.read_csv(config.TEST_PATH).iloc[0][config.FEATURE_NAMES].tolist()


def test_home_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"Rock or mine?" in res.data


def test_health(client):
    assert client.get("/api/health").get_json()["status"] == "ok"


def test_models_endpoint_lists_four_models(client):
    assert len(client.get("/api/models").get_json()["models"]) == 4


def test_samples_endpoint(client):
    data = client.get("/api/samples").get_json()
    assert len(data["samples"]) == 8
    assert len(data["class_means"]["M"]) == 60


def test_predict_ok(client):
    res = client.post("/api/predict", json={"features": _row(), "model": "knn"})
    assert res.status_code == 200
    assert res.get_json()["model"] == "knn"


def test_predict_rejects_bad_input(client):
    res = client.post("/api/predict", json={"features": [0.1, 0.2]})
    assert res.status_code == 400
    assert "60" in res.get_json()["error"]


def test_batch_reports_errors_per_row(client):
    res = client.post("/api/predict/batch", json={"readings": [_row(), [0.1]]})
    results = res.get_json()["results"]
    assert "label" in results[0] and "error" in results[1]


def test_cors_header_on_api(client):
    assert client.get("/api/health").headers["Access-Control-Allow-Origin"] == "*"
