#!/bin/bash

# Start Ray Pipeline Server with 8 GPUs
# This script sets up the environment and starts the advanced Ray pipeline server

set -e

echo "🚀 Starting Advanced Ray Pipeline Server..."

# Check if Ray is installed
if ! /opt/conda/envs/gizmo/bin/python -c "import ray" 2>/dev/null; then
    echo "❌ Ray is not installed. Installing Ray..."
    /opt/conda/envs/gizmo/bin/pip install "ray[serve]"
fi

# Set environment variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export CONDA_DEFAULT_ENV="gizmo"
export CONDA_PREFIX="/opt/conda/envs/gizmo"
export PATH="/opt/conda/envs/gizmo/bin:$PATH"

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --list-gpus

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "📊 Found $NUM_GPUS GPUs"

# Set default port
# Parse arguments
PORT=5555
MODEL="wan"

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      # Check if it's a number (port as positional argument for backward compatibility)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        PORT="$1"
        shift
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      ;;
  esac
done

echo "⚙️  Configuration:"
echo "   - Port: $PORT"
echo "   - GPUs: $NUM_GPUS"
echo "   - Model: $MODEL"
echo "   - Python path: $PYTHONPATH"

# Check if model is veo and GOOGLE_API_KEY is not set
if [[ "$MODEL" == "veo" && -z "$GOOGLE_API_KEY" ]]; then
    echo "❌ Error: GOOGLE_API_KEY is not set. Veo mode requires a valid API key."
    echo "   Please set it using: export GOOGLE_API_KEY='your_api_key'"
    exit 1
fi

# Create output directories
mkdir -p ./server_outputs
mkdir -p ./client_outputs

echo "📁 Created output directories"

# Start the server
echo "🎯 Starting server on port $PORT with $NUM_GPUS GPUs (Model: $MODEL)..."
/opt/conda/envs/gizmo/bin/python ray_pipeline_server.py --port $PORT --num-gpus $NUM_GPUS --model $MODEL 