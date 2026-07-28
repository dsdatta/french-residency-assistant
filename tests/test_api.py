import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_query_endpoint_returns_200():
    response = client.post("/query", json={"question": "What is civic exam ?"})
    assert response.status_code == 200


def test_query_response_has_required_fields():
    response = client.post("/query", json={"question": "What is civic exam ?"})
    data = response.json()
    assert "answer" in data
    assert "question" in data


def test_query_echoes_question():
    question = "What is the OFII medical visit?"
    response = client.post("/query", json={"question": question})
    assert response.json()["question"] == question


def test_empty_question_returns_200():
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 200


def test_missing_question_field_returns_422():
    response = client.post("/query", json={})
    assert response.status_code == 422
