# Hand Written Digit Classifier Numpy Only
The Handwritten Digit Classifier is a project developed to classify digits drawn by the user in an interface. The training and calculations of the algorithm were implemented using only the Numpy library. The main objective of this project is to gain an in-depth understanding of how a neural network functions. It aims to provide insights into the underlying mechanics of neural networks without relying on high-level libraries, focusing instead on core concepts like forward propagation, backpropagation, and gradient descent.

Richard Feynman: What I cannot create, I do not understand.

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



