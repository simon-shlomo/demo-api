"""import string
A simple model demo API

The API predicts the language of a text using a pre-trained model
"""
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.model.model import __version__ as model_version
from app.model.model import predict_pipeline

app = FastAPI()


class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str

@app.get("/", response_class=HTMLResponse)
def read_demo():
    """
    The root returns a basic demo form
    
    :return: Returns the demo html page
    """

    return generate_html_response()


@app.get("/info")
def home():
    """
    Returns basic information about the api
    
    :return: a JSON with healthcheck string and current model version 
    """

    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    """
    Predicts the language of the text

    :param payload: A JSON {"text": textToPredict}
    :return: the predicted language as a JSON {"language": predictedLanguage}
    """
    language = predict_pipeline(payload.text)
    return {"language": language}


@app.post("/submit")
def submit(text: str = Form()):
    """
    Accepts input from html form, and will predict the language of the entered text

    :param text: predicts the language of the "text" input field
    :return: the predicted language as JSON {"language": predictedLanguage}
    """
    language = predict_pipeline(text)
    return {"language": language}


def generate_html_response():
    """
    Generates the html of the root demo page

    :return: a basic html form
    """
    html_content = """
    <html>
        <head>
            <title>Simple demo-api</title>
        </head>
        <body>
            <h1>Simple demo-api: Predict language of text 
                <form action="/submit" method="post">
                    <label for="Text">Text to predict:</label><br>
                    <input type="text" id="text" name="text">
                    <input type="submit" value="Predict language">
                </form>	
        </body>
    </html>
    """
    return HTMLResponse(content=html_content, status_code=200)


#
# Tests of the code above - python3 -m pytest app/main.py
#
client = TestClient(app)


def test_read_info():
    """
    Tests that the /info endpoint is responding
    """
    response = client.get("/info")
    assert response.status_code == 200


def test_create_predict():
    """
    Tests that the /predict endpoint is predicting a simple text
    """
    response = client.post( "/predict/", json={"text": "Hello World"})
    assert response.status_code == 200
    assert response.json() == {
        "language": "English",
    }