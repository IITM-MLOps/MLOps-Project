# Hand Written Digit Classifier Numpy Only
The Handwritten Digit Classifier is a project developed to classify digits drawn by the user in an web interface. The training and calculations were implemented using only the Numpy library. 

Richard Feynman: What I cannot create, I do not understand.


Here’s a **summary** of how **MLflow** and **Spark** are used in your code, framed within your **MLOps project context**:

---

# Data Engineering [Spark used]

- **Data Ingestion / Transformation**:  
  You use **PySpark** (`SparkSession`) to load and preprocess the **MNIST images and labels**:
  - Images and labels are read from binary files using Python's `struct` and `numpy`.
  - The data is parallelized into **RDDs** and then converted into **Spark DataFrames** with explicit schemas (`StructType`).
  - This prepares the MNIST dataset in a distributed way, though the dataset is relatively small.

- **Throughput and Speed**:  
  Since MNIST is a lightweight dataset, Spark isn't fully utilized for scalability here. However, using Spark **shows readiness for scaling to bigger datasets** and **makes ingestion structured and parallelizable**.

- **Presence of Airflow/Spark/Custom pipeline**:  
  You are using **Spark** as the **data engineering pipeline** (✔️) — there is no Airflow involved here.

---

# Source Control & Continuous Integration [DVC and Git expected, but missing in your code]

- **DVC/CI/CD**:  
  In the code you shared, there is **no implementation of DVC** yet (e.g., no `dvc.yaml`, `dvc add` or DVC DAG structure).
  
- **Source/Model Versioning**:  
  You are using **pickle** (`save_model`, `load_model`) for **model persistence**, but there's no DVC/Git versioning shown explicitly.  
  You might need to add:
  - Git repo (`git init`)
  - Git LFS (for large files like models)
  - DVC (for tracking datasets and models)
  - A DVC pipeline (for CI-style stages like ingestion ➔ training ➔ evaluation).

---

# Experiment Tracking [MLflow used properly]

- **Experiment Tracking with MLflow**:
  - **Experiment** is set with `mlflow.set_experiment("MNIST_Digit_Classification")`.
  - **Tracking URI** is set to local filesystem (`file:./mlruns`).
  - You start a **run** (`with mlflow.start_run`) and **log parameters** (hyperparameters like learning rate, epochs, etc).
  - You **log metrics** (train and test accuracy).
  - You also **log artifacts manually** (you save a `metrics.json`, commented out logging artifacts though — needs `mlflow.log_artifact("metrics.json")` to activate).
  - You **set tags** to describe model type and dataset — beyond default autologging (✔️).

- **Tracked Items**:
  - **Parameters**: learning rate, number of epochs, batch sizes, layer sizes, output size.
  - **Metrics**: train accuracy, test accuracy.
  - **Artifacts** (partially): metrics.json (model file is saved, but not yet logged to MLflow).
  - **Tags**: model type ("Custom Neural Network"), dataset ("MNIST").

---

# 📌 Quick Checklist Mapping to Your Questions:
| Area                     | Status | Notes |
|---------------------------|--------|-------|
| Spark pipeline present?   | ✔️ | Data ingestion using Spark. |
| Airflow present?          | ❌ | Not used. |
| Throughput speed?         | ➡️ | MNIST is small; pipeline is ready for scalability though. |
| DVC or Git versioning?    | ❌ | Not shown yet; recommended to add. |
| Experiment tracking?      | ✔️ | Using MLflow manually (parameters, metrics, tags). |
| Tracking other info?      | ✔️ | Custom tags beyond autologging. |

---

Got it — you’ve now shared your **DVC pipeline** (`dvc.yaml`) for the project.  
Let’s **summarize and update** everything properly, including how this fits into your **MLOps structure** + how you can **connect it with GitHub Actions** for CI/CD.

---

# 📄 Updated DVC Pipeline (your `dvc.yaml`)
**Stages you have defined:**

| Stage Name          | Command | Key Actions |
|---------------------|---------|-------------|
| `preprocess`         | `python download_mnist.py` | Download MNIST data (saved into `mnist/`). |
| `train`              | `python train.py` | Train the model ➔ Output model `model_save_test.pkl` and `metrics.json`. |
| `serve`              | `nohup python fast.py > server.log 2>&1 &` | Start FastAPI server in background. |
| `test`               | `python test_api.py` | Test the FastAPI server by sending a request. Output: `test_result.json`. |
| `kill all process`   | `pkill uvicorn || true` | Kill Uvicorn server process after testing. |

---

# Key Updates and Notes:
- ✅ **Model versioning** is handled (`model_save_test.pkl` as `outs` in DVC).
- ✅ **Metrics tracking** is done (`metrics.json` tracked inside DVC).
- ✅ **Artifacts** like `server.log` and `test_result.json` are managed.


---

# 🧠 Summary: Your Updated MLOps Architecture
| Component | Tool | Status |
|-----------|------|--------|
| Data Ingestion | Spark | ✔️ |
| Model Training | Custom NN + MLflow tracking | ✔️ |
| Experiment Tracking | MLflow | ✔️ |
| Version Control | Git + DVC | ✔️  |
| Pipeline Automation | GitHub Actions (CI) | ✔️  |
| Model Serving | FastAPI | ✔️ |
| Model Testing | API-based tests | ✔️ |

---

# ⚡ Final small TODOs
- `dvc remote` is not configured (like GDrive, S3, etc.)

---

Would you also like me to show you an even **cleaner DAG diagram** of your DVC stages? 🧩  
It’ll help you visualize it if you need to show in your project report or presentation! 🎯  (I can generate it for you.)
<!-- https://github.com/user-attachments/assets/39132ce1-bf7e-4020-86cf-b6afc05fa541 -->

``
<img width="760" alt="Screenshot 2025-04-27 at 8 38 46 PM" src="https://github.com/user-attachments/assets/ccccb5bb-fd55-4be2-95b2-f15914c07092" />

<img width="1463" alt="Screenshot 2025-04-27 at 8 38 25 PM" src="https://github.com/user-attachments/assets/ec3d1a94-15f5-41fc-b415-e9c21240da8a" />

<img width="1099" alt="Screenshot 2025-04-26 at 6 03 01 PM" src="https://github.com/user-attachments/assets/dfa28d0b-4593-4545-8fe9-686d7673fbf7" />


# Doodle Digit Predictor User Manual

This guide helps non-technical users draw and predict digits using the web application.

## Getting Started
- **URL**: `http://localhost:3000/index.html`
- **Requirements**: A web browser (Chrome or Firefox recommended).

## How to Use
1. **Draw a Digit**:
   - Use your mouse to draw a digit (0–9) on the black canvas.
   - Click and drag to draw with a yellow brush.
2. **Predict the Digit**:
   - Click the "Predict Digit" button.
   - The predicted digit will appear below the canvas (e.g., "Digit: 5").
3. **Clear the Canvas**:
   - Click the "Clear Canvas" button to erase your drawing and start over.
4. **View Errors**:
   - If prediction fails, an error message will appear in red (e.g., "Connection Failed").

## Tips
- Draw clearly and fill the canvas for accurate predictions.
- Ensure the backend server is running (contact your administrator if errors occur).

## Troubleshooting
- **Canvas Not Responding**: Try a different browser or refresh the page.
- **Prediction Fails**: Check your internet connection or contact the administrator.
- **Error Messages**: Read the red text below the canvas for details.

<!-- ## Screenshot
[Insert screenshot of the canvas with a drawn digit and result] -->


## HLD Diagram

```mermaid
flowchart TD
    %% Subgraphs for logical grouping
    subgraph User_Interaction
        A[User] -->|Interacts| B[Frontend UI<br>HTML/JS]
    end
    subgraph Serving
        B -->|HTTP Requests| C[REST API<br>FastAPI]
        C -->|Predictions| D[Inference Engine<br>Numpy NN]
    end
    subgraph Monitoring
        C -->|Metrics| E[Prometheus<br>Monitoring]
        D -->|Metrics| E
        E -->|Visualize| G[Grafana<br>Dashboard]
        D -->|Experiment Data| F[MLflow<br>Tracking]
    end
    subgraph Training
        H[Training Data<br>MNIST] -->|Input| I[Training Module<br>Spark + Numpy]
        I -->|Stores| J[Model Artifact<br>MLflow]
        J -->|Loads| D
        F -->|Evaluation| K[Evaluation<br>Module]
        K -->|Triggers Retrain| I
    end

    %% Styling
    classDef User fill:#D4F4DD,stroke:#2E7D32,color:#2E7D32
    classDef UI fill:#FFF3E0,stroke:#F57C00,color:#F57C00
    classDef API fill:#E2EBFF,stroke:#374D7C,color:#374D7C
    classDef Inference fill:#FCE4EC,stroke:#C2185B,color:#C2185B
    classDef Monitoring fill:#E8F5E9,stroke:#388E3C,color:#388E3C
    classDef Data fill:#F3E5F5,stroke:#7B1FA2,color:#7B1FA2
    classDef Training fill:#E0F7FA,stroke:#0288D1,color:#0288D1
    classDef Evaluation fill:#FFF8E1,stroke:#FBC02D,color:#FBC02D

    A:::User
    B:::UI
    C:::API
    D:::Inference
    E:::Monitoring
    F:::Monitoring
    G:::Monitoring
    H:::Data
    I:::Training
    J:::Training
    K:::Evaluation

    %% Link styling
    linkStyle 0,1,2,3,4,5,6,7,8,9 stroke:#555,stroke-width:2px

```bash
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"

dvc add mnist/
git add mnist.dvc
git commit -m "Track MNIST dataset"
```


```
dvc dag
+------------+ 
| preprocess | 
+------------+ 
       *       
       *       
       *       
  +-------+    
  | train |    
  +-------+    
       *       
       *       
       *       
  +-------+    
  | serve |    
  +-------+    
       *       
       *       
       *       
   +------+    
   | test |    
   +------+    
       *       
       *       
       *       
    +-----+    
    | end |    
    +-----+   
```

```
scrape_configs:
  - job_name: "fastapi-app"
    scrape_interval: 1s
    static_configs:
      - targets: ["localhost:7000"]


  - job_name: "custom_exporter"
    scrape_interval: 5s
    static_configs:
      - targets: ["localhost:18000"]


  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
`



