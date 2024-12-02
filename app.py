from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

# Initialize FastAPI app
app = FastAPI()

# Allow all origins (you can specify specific origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specify your frontend domain here
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the trained model
model = tf.keras.models.load_model("deepDTA_model_v3.h5")

# Load tokenizer_drug
with open('tokenizer_drug.pkl', 'rb') as f:
    tokenizer_drug = pickle.load(f)

# Load tokenizer_protein
with open('tokenizer_protein.pkl', 'rb') as f:
    tokenizer_protein = pickle.load(f)


# Request model
class PredictionRequest(BaseModel):
    drug: str
    protein: str


@app.post("/predict/")
def predict_affinity(request: PredictionRequest):
    try:
        drug = request.drug
        protein = request.protein
        drug_sequence = tokenizer_drug.texts_to_sequences([drug])
        drug_sequence = pad_sequences(drug_sequence, truncating = "post", maxlen = 85)
        protein_sequence = tokenizer_protein.texts_to_sequences([protein])
        protein_sequence = pad_sequences(protein_sequence, truncating = "post", maxlen = 1200)
        prediction = model.predict([drug_sequence, protein_sequence]).item()

        return {"affinity": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error occurred: {str(e)}")
