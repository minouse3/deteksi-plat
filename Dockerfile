# Use Python 3.11-slim for a stable and lightweight environment
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only the bare minimum system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install necessary Python libraries for inference in one go
# We MUST install torch and torchvision together from the same CPU index 
# to avoid the "operator torchvision::nms does not exist" error.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    ultralytics \
    opencv-python-headless \
    numpy \
    scikit-learn \
    joblib

# Copy only the source code and necessary files for the app
COPY src/ /app/src/
COPY app.py /app/
COPY models/ /app/models/

# Create output directory
RUN mkdir -p /app/outputs/web_app

EXPOSE 7860

# Command to run the application
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
