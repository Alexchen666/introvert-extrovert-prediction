#!/bin/bash

# Exit on any error
set -e

# Print startup message
echo "Starting Streamlit application..."

# Check if data directory exists and warn if train.csv is missing
if [ ! -f "data/train.csv" ]; then
    echo "Warning: data/train.csv not found. Please ensure the data file is mounted or copied to the container."
    echo "The application may not work properly without the training data."
fi

# Activate the virtual environment created by uv
source .venv/bin/activate

# Start Streamlit with proper configuration for Docker
exec uv run streamlit run streamlit_app.py --browser.gatherUsageStats false \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false