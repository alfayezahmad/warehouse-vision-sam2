# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies for OpenCV and Git to clone SAM 2
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from your requirements list
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install SAM 2 directly from Meta's official repository
RUN pip install git+https://github.com/facebookresearch/segment-anything-2.git

# Copy the rest of the application
COPY . .

# Set environment variables for container paths
ENV DATA_DIR=/app/data
ENV DB_PATH=/app/data/robot_memory.db
ENV OUTPUT_PATH=/app/data/final_output.mp4
ENV MODEL_WEIGHT_PATH=/app/sam2_hiera_large.pt

# Run the pipeline
CMD ["python", "main_pipeline_v2.py"]
