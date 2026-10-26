import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402

SPAM = "WINNER!! You have won a 1000 prize. Call 09061701461 now to claim your free reward!"
HAM = "Hey, are we still meeting for lunch at 1pm tomorrow?"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_home_page_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"<textarea" in r.data


def test_health(client):
    assert client.get("/health").get_json() == {"status": "ok"}


def test_api_detects_spam(client):
    r = client.post("/api/predict", json={"message": SPAM})
    assert r.status_code == 200
    assert r.get_json()["prediction"] == "Spam"


def test_api_detects_not_spam(client):
    r = client.post("/api/predict", json={"message": HAM})
    assert r.status_code == 200
    assert r.get_json()["prediction"] == "Not Spam"


def test_api_confidence_range(client):
    conf = client.post("/api/predict", json={"message": SPAM}).get_json()["confidence"]
    assert 50 <= conf <= 100


def test_api_rejects_empty(client):
    assert client.post("/api/predict", json={"message": "  "}).status_code == 400
    assert client.post("/api/predict", json={}).status_code == 400


def test_api_rejects_too_long(client):
    assert client.post("/api/predict", json={"message": "a" * 1001}).status_code == 400


def test_form_spam_result(client):
    r = client.post("/", data={"message": SPAM})
    assert r.status_code == 200
    assert b"is-spam" in r.data


def test_form_not_spam_result(client):
    r = client.post("/", data={"message": HAM})
    assert b"is-safe" in r.data


def test_form_empty_shows_error(client):
    r = client.post("/", data={"message": ""})
    assert b"Enter a message" in r.data
