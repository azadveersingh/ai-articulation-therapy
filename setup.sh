#!/bin/bash

# Check if an environment name is provided
if [ -z "$1" ]; then
    echo "Error: Please provide an environment name."
    echo "Usage: $0 <environment_name>"
    exit 1
fi

ENV_NAME=$1
CUDA_VERSION="12.4"  # Adjust this if you have a different CUDA version installed
PYTHON_VERSION="3.10"  # Adjust this if you need a different Python version

# Function to check if a command succeeded
check_status() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed. Exiting."
        exit 1
    fi
}

# Update package list and install system dependencies
echo "Updating package list and installing system dependencies..."
sudo apt update
sudo apt install -y build-essential libgomp1
check_status "System dependencies installation"

# Ensure CUDA is in PATH (assuming CUDA is installed at /usr/local/cuda-<version>)
echo "Configuring CUDA environment..."
CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
if [ ! -d "$CUDA_PATH" ]; then
    echo "Error: CUDA ${CUDA_VERSION} not found at ${CUDA_PATH}. Please adjust CUDA_VERSION in the script."
    exit 1
fi
export PATH="${CUDA_PATH}/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/lib:/usr/lib/x86_64-linux-gnu:/lib:/lib/x86_64-linux-gnu:${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"

# Set system linker explicitly
export LD="/usr/bin/ld"

# Check if Conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed. Please install Anaconda/Miniconda first."
    exit 1
fi

# Create a new Conda environment
echo "Creating Conda environment: ${ENV_NAME} with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
check_status "Conda environment creation"

# Activate the environment
echo "Activating environment: ${ENV_NAME}..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
check_status "Environment activation"

# Install pip and other Python dependencies
echo "Installing pip and basic dependencies..."
conda install pip -y
check_status "Pip installation"

# Clean up previous llama-cpp-python installations and pip cache
echo "Cleaning up previous installations..."
pip uninstall llama-cpp-python -y 2>/dev/null
rm -rf ~/.cache/pip

# Install llama-cpp-python with CUDA support
echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_OPENMP=ON -DCMAKE_CXX_FLAGS=-fopenmp" pip install llama-cpp-python --no-cache-dir
check_status "llama-cpp-python installation"

echo "Installation completed successfully!"
echo "To activate the environment, run: conda activate ${ENV_NAME}"
