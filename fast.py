from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import numpy as np
import os
import pickle
import logging
import argparse
import json
from utils import *
from dense_neural_class import *

# Set up logging to debug issues in Codespaces
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predict Digits")

# Allow all origins for debugging in Codespaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Configurable port number
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7000)
args = parser.parse_args()
port = args.port

def load_model(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, filename)
    try:
        with open(filepath, 'rb') as file:
            model_loaded = pickle.load(file)
        logger.info(f"Model loaded successfully from {filepath}")
        return model_loaded
    except FileNotFoundError:
        logger.error(f"Model file {filepath} not found")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

# Load the model
try:
    model = load_model('model_save_test.pkl')
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

# Data drift detection setup
reference_stats_path = 'reference_stats.json'

# If reference stats file does not exist or is invalid, create it with MNIST dataset statistics
def initialize_reference_stats():
    reference_stats = {
        "mean": 0.1307,  # MNIST dataset mean pixel value
        "std": 0.3081    # MNIST dataset std deviation
    }
    try:
        with open(reference_stats_path, 'w') as f:
            json.dump(reference_stats, f, indent=2)
        logger.info(f"Created reference_stats.json with default MNIST stats at {reference_stats_path}")
        return reference_stats
    except Exception as e:
        logger.error(f"Failed to create reference_stats.json: {str(e)}")
        return reference_stats  # Return default stats even if write fails

# Load or initialize reference stats
if not os.path.exists(reference_stats_path):
    reference_stats = initialize_reference_stats()
else:
    try:
        with open(reference_stats_path, 'r') as f:
            content = f.read().strip()
            if not content:  # Check if file is empty
                logger.warning(f"reference_stats.json is empty, reinitializing with default stats")
                reference_stats = initialize_reference_stats()
            else:
                reference_stats = json.loads(content)
                logger.info(f"Loaded reference stats from {reference_stats_path}")
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in reference_stats.json: {str(e)}, reinitializing with default stats")
        reference_stats = initialize_reference_stats()
    except Exception as e:
        logger.error(f"Error reading reference_stats.json: {str(e)}, using default stats")
        reference_stats = initialize_reference_stats()

def detect_data_drift(new_data, threshold=0.1):
    """
    Detect data drift by comparing mean and std of new_data with reference stats.
    Returns True if drift detected, False otherwise.
    """
    new_mean = np.mean(new_data)
    new_std = np.std(new_data)

    mean_drift = abs(new_mean - reference_stats['mean'])
    std_drift = abs(new_std - reference_stats['std'])

    if mean_drift > threshold or std_drift > threshold:
        logger.warning(f"Data drift detected: mean_drift={mean_drift}, std_drift={std_drift}")
        return True
    return False

class Data(BaseModel):
    image_vector: list[float]

@app.post("/predict/")
async def upload_image(data: Data):
    try:
        # Validate input length (MNIST images are 28x28 = 784 pixels)
        if len(data.image_vector) != 784:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 784 values for MNIST image, got {len(data.image_vector)}"
            )
        # Reshape input for the model
        image_data = np.array(data.image_vector)
        logger.info(f"Input reshaped to shape: {image_data.shape}")

        # Data drift detection
        if detect_data_drift(image_data):
            logger.warning("Data drift detected in input data.")
            # Optionally, raise a warning or error or log for monitoring
            # For now, just log and continue

        # Make the prediction
        result = model.predict(image_data)[0]
        logger.info(f"Prediction result: {result}")
        return {"Result": int(result)}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
