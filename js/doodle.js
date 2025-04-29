document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('doodle-canvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const feedbackBtn = document.getElementById('feedback-btn');
    const result = document.getElementById('result');
    const error = document.getElementById('error');
    const feedbackStatus = document.getElementById('feedback-status');
    const actualDigitInput = document.getElementById('actual-digit');
    const feedbackSection = document.getElementById('feedback-section');

    let isDrawing = false;
    let lastImageVector = null;
    let lastPredictedDigit = null;

    // Drawing settings
    ctx.lineWidth = 20; // Matches brush_size in Tkinter app
    ctx.strokeStyle = 'yellow'; // Visible on black background
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Mouse events for drawing
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        ctx.beginPath();
        ctx.moveTo(e.offsetX, e.offsetY);
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDrawing) {
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
        }
    });

    canvas.addEventListener('mouseup', () => {
        isDrawing = false;
        ctx.closePath();
    });

    canvas.addEventListener('mouseout', () => {
        isDrawing = false;
        ctx.closePath();
    });

    // Predict button
    predictBtn.addEventListener('click', async () => {
        error.textContent = ''; // Clear previous errors
        feedbackStatus.textContent = ''; // Clear previous feedback status
        result.textContent = 'Predicting...';
        feedbackSection.style.display = 'none'; // Hide feedback section initially

        // Convert canvas to 28x28 grayscale image
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28); // Resize to 28x28
        const imageData = tempCtx.getImageData(0, 0, 28, 28);
        const pixels = imageData.data;

        // Convert to grayscale and normalize (0 to 1)
        const imageVector = new Array(784).fill(0);
        for (let i = 0; i < pixels.length; i += 4) {
            const grayscale = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3; // Average RGB
            imageVector[i / 4] = grayscale / 255; // Normalize
        }

        // Store the image vector for potential feedback
        lastImageVector = imageVector;

        // Send to FastAPI
        try {
            const response = await fetch("http://localhost:7000/predict", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_vector: imageVector })
            });
            if (response.ok) {
                const data = await response.json();
                lastPredictedDigit = data.Result;
                result.textContent = `Digit: ${data.Result}`;
                feedbackSection.style.display = 'block'; // Show feedback section after prediction
            } else {
                error.textContent = `API Error: ${response.statusText}`;
                result.textContent = 'Prediction failed.';
                lastPredictedDigit = null;
                feedbackSection.style.display = 'none'; // Hide feedback section on error
            }
        } catch (err) {
            error.textContent = `Connection Failed: ${err.message}`;
            result.textContent = 'Prediction failed.';
            lastPredictedDigit = null;
            feedbackSection.style.display = 'none'; // Hide feedback section on error
        }
    });

    // Feedback button
    feedbackBtn.addEventListener('click', async () => {
        feedbackStatus.textContent = ''; // Clear previous feedback status
        error.textContent = ''; // Clear previous errors

        if (!lastImageVector || lastPredictedDigit === null) {
            feedbackStatus.textContent = 'No prediction available to provide feedback on. Please predict a digit first.';
            return;
        }

        const actualDigitValue = actualDigitInput.value.trim();
        if (actualDigitValue !== '') {
            const actualDigit = parseInt(actualDigitValue);
            if (!isNaN(actualDigit) && actualDigit >= 0 && actualDigit <= 9) {
                // Send feedback to backend
                try {
                    const response = await fetch('http://localhost:7000/feedback', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image_vector: lastImageVector,
                            predicted_digit: lastPredictedDigit,
                            actual_digit: actualDigit
                        })
                    });
                    if (response.ok) {
                        const feedbackData = await response.json();
                        feedbackStatus.textContent = feedbackData.message || 'Feedback submitted successfully!';
                        // Optionally, hide feedback section after submission
                        feedbackSection.style.display = 'none';
                        actualDigitInput.value = ''; // Clear input field
                    } else {
                        error.textContent = `Feedback API Error: ${response.statusText}`;
                    }
                } catch (err) {
                    error.textContent = `Feedback Connection Failed: ${err.message}`;
                }
            } else {
                feedbackStatus.textContent = 'Invalid actual digit input. Please enter a number between 0 and 9.';
            }
        } else {
            feedbackStatus.textContent = 'Please enter the actual digit before submitting feedback.';
        }
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        result.textContent = 'Draw a digit and click "Predict Digit" to see the result.';
        error.textContent = '';
        feedbackStatus.textContent = '';
        lastImageVector = null;
        lastPredictedDigit = null;
        actualDigitInput.value = ''; // Clear the input field
        feedbackSection.style.display = 'none'; // Hide feedback section
    });
});
