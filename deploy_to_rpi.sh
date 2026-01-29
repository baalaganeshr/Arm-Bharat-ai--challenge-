#!/bin/bash

# ============================================
# Deployment Script for Raspberry Pi
# Face Mask Detection with FPGA Acceleration
# ============================================

echo "========================================="
echo "  Face Mask Detection - RPi Deployment"
echo "========================================="

# Configuration - UPDATE THESE VALUES
RPI_HOST="pi@192.168.1.100"     # Change to your RPi IP address
PROJECT_DIR="/home/pi/mask-detection"
RPI_PASSWORD=""                  # Optional: set password for sshpass

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if SSH is available
if ! command -v ssh &> /dev/null; then
    print_error "SSH is not installed. Please install OpenSSH."
    exit 1
fi

# Check connection to RPi
print_status "Testing connection to Raspberry Pi..."
if ! ssh -o ConnectTimeout=5 -o BatchMode=yes $RPI_HOST "echo 'Connected'" 2>/dev/null; then
    print_warning "Could not connect automatically. You may need to enter password."
fi

# Create remote directory
print_status "Creating remote directory..."
ssh $RPI_HOST "mkdir -p $PROJECT_DIR/{modified,our_improvements,models,logs,dataset}"

# Copy Python files
print_status "Copying modified/ folder..."
scp -r modified/*.py $RPI_HOST:$PROJECT_DIR/modified/

print_status "Copying our_improvements/ folder..."
scp -r our_improvements/*.py $RPI_HOST:$PROJECT_DIR/our_improvements/
scp -r our_improvements/templates $RPI_HOST:$PROJECT_DIR/our_improvements/ 2>/dev/null || true

# Copy model files
print_status "Copying trained models..."
scp models/mask_detector_64x64.h5 $RPI_HOST:$PROJECT_DIR/models/
scp models/mask_detector_64x64.tflite $RPI_HOST:$PROJECT_DIR/models/ 2>/dev/null || true

# Copy requirements
print_status "Copying requirements.txt..."
scp requirements.txt $RPI_HOST:$PROJECT_DIR/

# Copy documentation
print_status "Copying documentation..."
scp docs/README.md $RPI_HOST:$PROJECT_DIR/ 2>/dev/null || true

# Install dependencies on RPi
print_status "Installing dependencies on Raspberry Pi..."
ssh $RPI_HOST << 'EOF'
    cd ~/mask-detection
    
    # Check if pip3 is available
    if ! command -v pip3 &> /dev/null; then
        echo "Installing pip3..."
        sudo apt-get update
        sudo apt-get install -y python3-pip
    fi
    
    # Install TensorFlow Lite (lighter for RPi)
    pip3 install tflite-runtime 2>/dev/null || pip3 install tensorflow
    
    # Install other dependencies
    pip3 install opencv-python-headless numpy pyserial pandas matplotlib flask
    
    # Create logs directory
    mkdir -p logs
    
    echo "Dependencies installed successfully!"
EOF

# Set permissions
print_status "Setting permissions..."
ssh $RPI_HOST "chmod +x $PROJECT_DIR/modified/*.py $PROJECT_DIR/our_improvements/*.py"

# Create run script on RPi
print_status "Creating run scripts on RPi..."
ssh $RPI_HOST << EOF
cat > $PROJECT_DIR/run_detection.sh << 'SCRIPT'
#!/bin/bash
cd ~/mask-detection
python3 modified/detect_fpga_simple.py
SCRIPT
chmod +x $PROJECT_DIR/run_detection.sh

cat > $PROJECT_DIR/run_dashboard.sh << 'SCRIPT'
#!/bin/bash
cd ~/mask-detection
python3 our_improvements/dashboard_app.py
SCRIPT
chmod +x $PROJECT_DIR/run_dashboard.sh
EOF

echo ""
echo "========================================="
echo -e "${GREEN}  Deployment Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps on Raspberry Pi:"
echo "  1. SSH into RPi: ssh $RPI_HOST"
echo "  2. Navigate to project: cd $PROJECT_DIR"
echo "  3. Connect FPGA via USB (check /dev/ttyUSB0)"
echo "  4. Run detection: ./run_detection.sh"
echo "  5. Or run dashboard: ./run_dashboard.sh"
echo ""
echo "FPGA Port Configuration:"
echo "  - Linux/RPi: /dev/ttyUSB0"
echo "  - Update in: modified/detect_fpga_simple.py"
echo ""
print_status "Done!"
