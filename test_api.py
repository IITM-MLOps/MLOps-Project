# This script just check api calls and does some light inferencing
import requests
import json
import time
import random
import os
from datetime import datetime

def test_api(num_attempts=3, delay_between_attempts=2):
    """
    Test the FastAPI server by sending prediction and feedback requests multiple times.
    
    Args:
        num_attempts (int): Number of times to attempt the prediction and feedback cycle.
        delay_between_attempts (int): Delay in seconds between each attempt.
    """
    print(f"Starting API tests with {num_attempts} attempts, {delay_between_attempts}s delay between attempts.")
    
    # Initialize results storage
    prediction_results = []
    feedback_results = []
    
    # Wait briefly to ensure server is up
    time.sleep(5)
    
    for attempt in range(1, num_attempts + 1):
        print(f"\nAttempt {attempt}/{num_attempts}:")
        
        # Create a test input vector of 784 values (for MNIST image)
        # Simulate different inputs by adding some random noise for variety
        image_vector = [random.uniform(0.0, 0.1) for _ in range(784)]
        
        # Test 1: Send POST request to the prediction API
        predict_url = "http://localhost:7000/predict/"
        headers = {
            "accept": "application/json",
            "Content-Type": "application/json"
        }
        predict_payload = {"image_vector": image_vector}
        
        try:
            predict_response = requests.post(predict_url, json=predict_payload, headers=headers, timeout=10)
            predict_response_data = predict_response.json() if predict_response.ok else {"error": predict_response.text}
            predict_status = predict_response.status_code
            print(f"Prediction test result: Status {predict_status}")
        except requests.exceptions.RequestException as e:
            predict_response_data = {"error": str(e)}
            predict_status = "connection_failed"
            print(f"Prediction test failed: {str(e)}")
        
        # Save individual prediction result
        predict_result = {
            "attempt": attempt,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status_code": predict_status,
            "response": predict_response_data
        }
        prediction_results.append(predict_result)
        
        # If prediction was successful, test the feedback endpoint
        if predict_status == 200 and "Result" in predict_response_data:
            predicted_digit = predict_response_data["Result"]
            # Simulate user feedback by randomly deciding if the prediction was correct
            # 70% chance of correct prediction, 30% chance of incorrect for testing variety
            is_correct = random.random() < 0.7
            if is_correct:
                actual_digit = predicted_digit
            else:
                actual_digit = random.choice([d for d in range(10) if d != predicted_digit])
            
            feedback_url = "http://localhost:7000/feedback/"
            feedback_payload = {
                "image_vector": image_vector,
                "predicted_digit": predicted_digit,
                "actual_digit": actual_digit
            }
            try:
                feedback_response = requests.post(feedback_url, json=feedback_payload, headers=headers, timeout=10)
                feedback_response_data = feedback_response.json() if feedback_response.ok else {"error": feedback_response.text}
                feedback_status = feedback_response.status_code
                print(f"Feedback test result: Status {feedback_status}")
            except requests.exceptions.RequestException as e:
                feedback_response_data = {"error": str(e)}
                feedback_status = "connection_failed"
                print(f"Feedback test failed: {str(e)}")
            
            # Save individual feedback result
            feedback_result = {
                "attempt": attempt,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "status_code": feedback_status,
                "response": feedback_response_data,
                "predicted_digit": predicted_digit,
                "actual_digit": actual_digit,
                "was_correct": is_correct
            }
            feedback_results.append(feedback_result)
        else:
            print("Skipping feedback test due to failed prediction.")
            feedback_results.append({
                "attempt": attempt,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "status_code": "skipped",
                "response": {"error": "Prediction failed, feedback not sent"},
                "predicted_digit": None,
                "actual_digit": None,
                "was_correct": False
            })
        
        # # Save intermediate results after each attempt
        # with open(f"test_result_attempt_{attempt}.json", "w") as f:
        #     json.dump(predict_result, f, indent=2)
        # if feedback_results[-1]["status_code"] != "skipped":
        #     with open(f"feedback_result_attempt_{attempt}.json", "w") as f:
        #         json.dump(feedback_results[-1], f, indent=2)
        
        # Delay before next attempt to avoid overwhelming the server
        if attempt < num_attempts:
            time.sleep(delay_between_attempts)

    # Save consolidated results
    consolidated_results = {
        "prediction_results": prediction_results,
        "feedback_results": feedback_results,
        "total_attempts": num_attempts,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    with open("test_result.json", "w") as f:
        json.dump(consolidated_results, f, indent=2)
    print(f"\nAll test results saved to test_result.json with {num_attempts} attempts completed.")

    # Generate and print summary
    total_predictions = len(prediction_results)
    successful_predictions = sum(1 for r in prediction_results if r['status_code'] == 200)
    total_feedbacks = len([f for f in feedback_results if f['status_code'] == 200])
    correct_feedbacks = sum(1 for f in feedback_results if f.get('was_correct'))
    incorrect_feedbacks = total_feedbacks - correct_feedbacks
    accuracy_estimate = (correct_feedbacks / total_feedbacks * 100) if total_feedbacks > 0 else "N/A"

    print("\nTest Run Summary:")
    print(f"Total Prediction Attempts: {total_predictions}")
    print(f"Successful Predictions: {successful_predictions} ({successful_predictions/total_predictions*100:.2f}%)")
    print(f"Total Feedback Submissions: {total_feedbacks}")
    print(f"Correct Feedbacks: {correct_feedbacks}")
    print(f"Incorrect Feedbacks: {incorrect_feedbacks}")
    print(f"Estimated Accuracy (from feedback): {accuracy_estimate:.2f}% if total_feedbacks > 0 else 'N/A'")

if __name__ == "__main__":
    # Run the test with multiple attempts for robustness
    test_api(num_attempts=3, delay_between_attempts=2)
