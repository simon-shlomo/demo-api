from fastapi import FastAPI
from pydantic import BaseModel
from app.model.model import predict_pipeline
from app.model.model import __version__ as model_version
from fastapi.testclient import TestClient


app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}

client = TestClient(app)


def test_read_home():
    response = client.get("/")
    assert response.status_code == 200

def test_create_predict():
    response = client.post(
        "/predict/",
        json={"text": "Hello World"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "language": "English",
    }
    
