from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Define input schema
class URLFeatures(BaseModel):
    url_length: int
    number_of_dots_in_url: int
    having_repeated_digits_in_url: int
    number_of_special_char_in_url: int
    entropy_of_url: float
    entropy_of_domain: float

app = FastAPI()

# Allow requests from your Vercel frontend
origins = [
    "https://phishing-detect-frontend.vercel.app",  # Your frontend domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(features: URLFeatures):
    input_data = np.array([[
        features.url_length,
        features.number_of_dots_in_url,
        features.having_repeated_digits_in_url,
        features.number_of_special_char_in_url,
        features.entropy_of_url,
        features.entropy_of_domain
    ]])
    prediction = model.predict(input_data)[0]
    return {"result": "Legitimate" if prediction == 0 else "Phishing"}
