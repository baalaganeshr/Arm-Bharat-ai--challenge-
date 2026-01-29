# Face Mask Detection with FPGA Acceleration
# Docker container for development environment

FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional packages for FPGA interface
RUN pip install --no-cache-dir \
    pyserial \
    pandas \
    matplotlib \
    scikit-learn

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p dataset/with_mask dataset/without_mask models logs docs

# Default command
CMD ["bash"]
