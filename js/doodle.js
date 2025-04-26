document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('doodle-canvas');
    const ctx = canvas.getContext('2d');
    const predictBtn = document.getElementById('predict-btn');
    const clearBtn = document.getElementById('clear-btn');
    const result = document.getElementById('result');
    const error = document.getElementById('error');

    let isDrawing = false;

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
        result.textContent = 'Predicting...';

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

        // Send to FastAPI
        try {
            const response = await fetch("https://curly-pancake-v6rjp5ggvqgwhwpwr-7000.app.github.dev/predict/", {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image_vector: imageVector })
            });
            if (response.ok) {
                const data = await response.json();
                result.textContent = `Digit: ${data.Result}`;
            } else {
                error.textContent = `API Error: ${response.statusText}`;
                result.textContent = 'Prediction failed.';
            }
        } catch (err) {
            error.textContent = `Connection Failed: ${err.message}`;
            result.textContent = 'Prediction failed.';
        }
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        result.textContent = 'Draw a digit and click "Predict Digit" to see the result.';
        error.textContent = '';
    });
});