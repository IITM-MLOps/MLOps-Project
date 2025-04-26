# test_api.py
import requests
import json
import time

def test_api():
    """Test the FastAPI server by sending a prediction request."""
    # Wait briefly to ensure server is up
    time.sleep(5)
    
    # Create a test input vector of 784 zeros (for MNIST image)
    image_vector = [0.0] * 784
    
    # Send POST request to the API
    url = "http://localhost:7000/predict/"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {"image_vector": image_vector}
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response_data = response.json() if response.ok else {"error": response.text}
        status = response.status_code
    except requests.exceptions.RequestException as e:
        response_data = {"error": str(e)}
        status = "connection_failed"
    
    # Save the result
    result = {
        "status_code": status,
        "response": response_data
    }
    with open("test_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Test result saved to test_result.json with status {status}")

if __name__ == "__main__":
    test_api()
