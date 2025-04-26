# Dockerfile
FROM python:3.10-slim

WORKDIR /Users/aayus/HandwrittenDigitClassifier

# Install system dependencies for OpenCV and Tkinter
RUN apt-get update && apt-get install -y 


# 3. Upgrade pip before installation
RUN python -m pip install --upgrade pip

# Copy requirements first to leverage Docker cache

# Copy requirements
COPY requirements.txt . 

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Environment variable for port with default value
EXPOSE 7000


CMD ["python","fast.py","--port","7000"]

