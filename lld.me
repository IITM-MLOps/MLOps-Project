
# Low-Level Design (LLD) Document for Doodle Digit Classifier
#### 1. Overview
This LLD document outlines the detailed implementation of the Handwritten Digit Classifier, a system I’ve built to predict digits (0-9) drawn on a web canvas using a custom **Numpy**-based neural network. Hosted at [aayushmanda/FastAPI](https://github.com/aayushmanda/FastAPI), the project integrates a web frontend, FastAPI backend, and monitoring tools, with a focus on modularity and clarity for maintenance and extension.

#### 2. System Components
- **Frontend UI (HTML/JS)**: A web canvas at `http://localhost:3000` for drawing digits, sending 784-pixel vectors to the backend.
- **Backend API (FastAPI)**: Hosted at `http://localhost:7000`, handles prediction and feedback requests.
- **Inference Engine (Numpy NN)**: Custom neural network for digit prediction, loaded from `model_save_test.pkl`.
- **Training Module (Spark + Numpy)**: Processes MNIST data with Spark, trains model with Numpy.
- **Monitoring (Prometheus/Grafana)**: Tracks metrics via `/metrics`, visualized at `http://localhost:3001`.

#### 3. Design Paradigm
**Does the software follow OO or Functional paradigm?**  
I’ve designed the software primarily using an **Object-Oriented (OO) paradigm**. The codebase, including `Dense_Neural_Diy` class in training and FastAPI app structure, relies on classes and objects for modularity and state management (e.g., model weights). This aligns with OO principles like encapsulation and reusability, though some data processing with Spark incorporates functional elements (e.g., transformations).

#### 4. API Endpoint Definitions with I/O Specifications
**Does the low-level design (LLD) document clearly specify the API endpoint definitions with the respective I/O specifications?**  
Yes, below are the detailed API endpoint definitions for my FastAPI backend, ensuring precise input/output (I/O) specifications as per the rubric.

| **Endpoint**       | **Method** | **URL**                          | **Input**                                                                 | **Output**                              | **Description**                          |
|---------------------|------------|----------------------------------|---------------------------------------------------------------------------|-----------------------------------------|------------------------------------------|
| `/predict/`         | POST       | `http://localhost:7000/predict/` | JSON: `{"image_vector": [float, ...]}` (array of 784 floats, 0.0-1.0)   | JSON: `{"Result": int}` (digit 0-9)    | Predicts digit from drawn image vector.  |
| `/feedback/`        | POST       | `http://localhost:7000/feedback/`| JSON: `{"image_vector": [float, ...], "predicted_digit": int, "actual_digit": int}` | JSON: `{"message": string}`            | Logs feedback on prediction accuracy.    |
| `/healthz`          | GET        | `http://localhost:7000/healthz`  | None                                                                     | JSON: `{"status": "ok"}`               | Checks API server health.                |
| `/metrics`          | GET        | `http://localhost:7000/metrics`  | None                                                                     | Text (Prometheus format): Metrics data | Exposes metrics for Prometheus scraping. |

- **Input Validation**: For `/predict/`, `image_vector` must be exactly 784 elements; else, returns 400 with error message (e.g., `{"detail": "Expected 784 values, got X"}`). For `/feedback/`, all fields are required; missing data returns 400 or 500.
- **Error Handling**: Exceptions in `/predict/` or `/feedback/` return 500 with details (e.g., `{"detail": "Prediction failed: "}`).

#### 5. Data Structures
- **Input (Frontend to Backend)**: `image_vector` as 784 floats (normalized 0.0-1.0) for 28x28 grayscale image.
- **Internal (Model)**: Numpy arrays for weights, biases, activations (input: 784, hidden: 50/20, output: 10).
- **Output (Backend to Frontend)**: Integer digit (0-9) or error message in JSON.

#### 6. File Structure
- **Frontend**: `frontend/index.html`, `frontend/js/doodle.js`, `frontend/css/styles.css`.
- **Backend**: `fast.py`, `utils.py`, `dense_neural_class.py`.
- **Models**: `models/model_save_test_.pkl`.
- **Logs**: `inference_metrics.json`, `feedback_log.json`.

#### 7. Interactions
- **Frontend-Backend**: Sends POST to `/predict/` with image data, receives digit; sends POST to `/feedback/` with feedback.
- **Backend-Model**: Loads model, processes input via `predict()`, logs metrics.
- **Backend-Monitoring**: Exposes metrics at `/metrics` for Prometheus, scraped for Grafana.

