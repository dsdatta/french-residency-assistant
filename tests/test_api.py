import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)


def get_auth_headers():
    response = client.post("/token", data={"username": "sameer", "password": "secret"})
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "OK"}


def test_query_endpoint_returns_200():
    response = client.post(
        "/query", json={"question": "What is civic exam ?"}, headers=get_auth_headers()
    )
    assert response.status_code == 200


def test_query_response_has_required_fields():
    response = client.post(
        "/query", json={"question": "What is civic exam ?"}, headers=get_auth_headers()
    )
    data = response.json()
    assert "answer" in data
    assert "question" in data


def test_query_echoes_question():
    question = "What is the OFII medical visit?"
    response = client.post(
        "/query", json={"question": question}, headers=get_auth_headers()
    )
    assert response.json()["question"] == question


def test_empty_question_returns_200():
    response = client.post("/query", json={"question": ""}, headers=get_auth_headers())
    assert response.status_code == 200


def test_missing_question_field_returns_422():
    response = client.post("/query", json={})
    assert response.status_code == 422
