#!/usr/bin/python

#!pip install spacy fastapi

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
from utils import *
from dense_neural_class import *
import os
import pickle
# create the webapp.
app = FastAPI(title="Predict Digits")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["https://curly-pancake-v6rjp5ggvqgwhwpwr-3000.app.github.dev"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#configurable portnumber
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7000)
args = parser.parse_args()
port = args.port

def load_model(filename):
    # Gets the current directory where the script is being executed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Constructs the full path of the .pkl file
    filepath = os.path.join(current_dir, filename)
    
    with open(filepath, 'rb') as file:
        model_loaded = pickle.load(file)
    
    return model_loaded

class Data(BaseModel):
    image_vector:list[float]
    # lang:str

model = load_model('model.pkl')

@app.post("/predict/")
async def upload_image(data: Data):
    image_data = np.array(data.image_vector)

    # Make the prediction
    result = model.predict(image_data)[0]
    print(result)
    return {"Result": int(result)}



uvicorn.run(app, host='0.0.0.0', port=port)