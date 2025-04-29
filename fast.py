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
import glob
import subprocess
from datetime import datetime
from utils import *
from dense_neural_class import *
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, REGISTRY

# Set up logging to debug issues
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Predict Digits")

# Allow all origins for debugging
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

# Initialize Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, endpoint="/metrics")

# Custom Prometheus metrics (explicitly registered)
data_drift_counter = Counter(
    'data_drift_detected_total',
    'Total number of data drift detections',
    registry=REGISTRY
)
api_call_counter = Counter(
    'api_predict_calls_total',
    'Total number of calls to the predict endpoint',
    registry=REGISTRY
)
prediction_error_counter = Counter(
    'prediction_error_total',
    'Total number of reported prediction errors',
    registry=REGISTRY
)

# File to store cumulative data drift and error counts
METRICS_LOG_FILE = 'inference_metrics.json'

def update_metrics_log(data_drift_increment=0, error_increment=0):
    """
    Update the cumulative metrics log file with data drift and error counts.
    """
    metrics = {"data_drift_count": 0, "prediction_error_count": 0, "last_updated": ""}
    if os.path.exists(METRICS_LOG_FILE):
        try:
            with open(METRICS_LOG_FILE, 'r') as f:
                metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics log: {str(e)}. Resetting metrics.")
    
    metrics["data_drift_count"] += data_drift_increment
    metrics["prediction_error_count"] += error_increment
    metrics["last_updated"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        with open(METRICS_LOG_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Updated metrics log: {metrics}")
    except Exception as e:
        logger.error(f"Error updating metrics log: {str(e)}")

def check_evaluation_thresholds():
    """
    Check if data drift or error counts exceed thresholds to trigger evaluation.
    If thresholds are exceeded, trigger evaluate_model.py.
    """
    DATA_DRIFT_THRESHOLD = 10  # Trigger evaluation if data drift count exceeds this
    ERROR_THRESHOLD = 5        # Trigger evaluation if error count exceeds this
    
    if os.path.exists(METRICS_LOG_FILE):
        try:
            with open(METRICS_LOG_FILE, 'r') as f:
                metrics = json.load(f)
                data_drift_count = metrics.get("data_drift_count", 0)
                error_count = metrics.get("prediction_error_count", 0)
                
                if data_drift_count > DATA_DRIFT_THRESHOLD or error_count > ERROR_THRESHOLD:
                    logger.info(f"Threshold exceeded: Data Drift={data_drift_count}, Errors={error_count}. Triggering evaluation.")
                    trigger_evaluation()
                else:
                    logger.info(f"Thresholds not exceeded: Data Drift={data_drift_count}, Errors={error_count}.")
        except Exception as e:
            logger.error(f"Error checking evaluation thresholds: {str(e)}")
    else:
        logger.info("Metrics log not found. Skipping evaluation check.")

def trigger_evaluation():
    """
    Trigger the evaluate_model.py script to assess if retraining is needed.
    """
    try:
        result = subprocess.run(['python3', 'evaluate_model.py'], capture_output=True, text=True)
        logger.info("Evaluation script output:")
        logger.info(result.stdout)
        if result.returncode != 0:
            logger.error(f"Evaluation script failed with error: {result.stderr}")
        else:
            logger.info("Evaluation script executed successfully")
    except FileNotFoundError:
        logger.error("evaluate_model.py script not found. Cannot trigger evaluation.")
    except Exception as e:
        logger.error(f"Error triggering evaluation script: {str(e)}")

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

def find_latest_pkl_file(directory='models'):
    """
    Find the latest .pkl file in the specified directory based on creation time.
    Args:
        directory (str): The directory to search for .pkl files.
    Returns:
        str or None: Path to the latest .pkl file, or None if no files are found.
    """
    try:
        list_of_files = glob.glob(os.path.join(directory, 'model_save_test_*.pkl'))
        if not list_of_files:
            logger.warning(f"No model files found in {directory} matching 'model_save_test_*.pkl'")
            return None
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f"Found latest model file: {latest_file}")
        return latest_file
    except Exception as e:
        logger.error(f"Error finding latest model file in {directory}: {str(e)}")
        return None

# Load the model
try:
    model_path = find_latest_pkl_file()
    if model_path and os.path.exists(model_path):
        model = load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
    else:
        fallback_path = 'model_save_test.pkl'
        if os.path.exists(fallback_path):
            model = load_model(fallback_path)
            logger.info(f"Loaded fallback model from {fallback_path}")
        else:
            raise FileNotFoundError("No model file found in the directory or fallback location")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

# Data drift detection setup
reference_stats_path = 'reference_stats.json'

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
        return reference_stats

if not os.path.exists(reference_stats_path):
    reference_stats = initialize_reference_stats()
else:
    try:
        with open(reference_stats_path, 'r') as f:
            content = f.read().strip()
            if not content:
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
    new_mean = np.mean(new_data)
    new_std = np.std(new_data)
    mean_drift = abs(new_mean - reference_stats['mean'])
    std_drift = abs(new_std - reference_stats['std'])
    if mean_drift > threshold or std_drift > threshold:
        logger.warning(f"Data drift detected: mean_drift={mean_drift}, std_drift={std_drift}")
        data_drift_counter.inc()
        update_metrics_log(data_drift_increment=1)  # Increment data drift count in log
        return True
    return False

class Data(BaseModel):
    image_vector: list[float]

class Feedback(BaseModel):
    image_vector: list[float]
    predicted_digit: int
    actual_digit: int

@app.post("/predict/")
async def upload_image(data: Data):
    try:
        api_call_counter.inc()
        if len(data.image_vector) != 784:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 784 values for MNIST image, got {len(data.image_vector)}"
            )
        image_data = np.array(data.image_vector)
        logger.info(f"Input reshaped to shape: {image_data.shape}")
        if detect_data_drift(image_data):
            logger.warning("Data drift detected in input data.")
        result = model.predict(image_data)[0]
        logger.info(f"Prediction result: {result}")
        # Check evaluation thresholds after each prediction (simulating periodic check)
        check_evaluation_thresholds()
        return {"Result": int(result)}
    except HTTPException as he:
        update_metrics_log(error_increment=1)  # Increment error count on exception
        check_evaluation_thresholds()
        raise he
    except Exception as e:
        update_metrics_log(error_increment=1)  # Increment error count on exception
        check_evaluation_thresholds()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/feedback/")
async def submit_feedback(feedback: Feedback):
    """
    Endpoint for users to report incorrect predictions as feedback.
    This can contribute to the error count for evaluation.
    """
    try:
        prediction_error_counter.inc()
        update_metrics_log(error_increment=1)  # Increment error count on feedback
        logger.info(f"Feedback received: Predicted {feedback.predicted_digit}, Actual {feedback.actual_digit}")
        # Optionally, log feedback details for further analysis
        feedback_log = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "predicted_digit": feedback.predicted_digit,
            "actual_digit": feedback.actual_digit,
            "image_vector": feedback.image_vector
        }
        with open('feedback_log.json', 'a') as f:
            json.dump(feedback_log, f)
            f.write('\n')  # Append newline for JSONL format
        check_evaluation_thresholds()
        return {"message": "Feedback received, thank you!"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run(app, host='0.0.0.0', port=port)
