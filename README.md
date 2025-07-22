# Introvert Extrovert Prediction

The work of Kaggle competition - [Predict the Introverts from the Extroverts Playground Series - Season 5, Episode 7](https://www.kaggle.com/competitions/playground-series-s5e7/overview)

This project includes:

1. Data preprocessing: EDA, Missing Value Imputation, and Feature Engineering.
2. Modelling.
3. Containerisation: Use Docker to build the replicable environment.
4. Visualisation: Use Streamlit to present the results and provide the function that can batch upload csv prediction and download the result.

## Docker Setup for Introvert-Extrovert Prediction App

This document provides instructions for building and running the Streamlit application using Docker.

### Prerequisites

- Docker (and docker-buildx) installed on your system
- Training data file (`data/train.csv`) - see Data Requirements section

### Files Created

- **Dockerfile**: Multi-stage Docker build configuration
- **entrypoint.sh**: Container startup script that configures and runs Streamlit

### Building the Docker Image

```bash
docker build -t introvert-extrovert-prediction:latest .
```

### Running the Container

```bash
docker run --rm -p 8501:8501 introvert-extrovert-prediction:latest
```

### Accessing the Application

Once the container is running, access the Streamlit app at:
- **URL**: http://localhost:8501
- **Port**: 8501

### Data Requirements

The application expects a CSV file at `data/train.csv` with the following columns:
- `id`: Unique identifier
- `Time_spent_Alone`: Hours spent alone daily (0-11)
- `Stage_fear`: Presence of stage fright (Yes/No)
- `Social_event_attendance`: Frequency of social events (0-10)
- `Going_outside`: Frequency of going outside (0-7)
- `Drained_after_socializing`: Feeling drained after socializing (Yes/No)
- `Friends_circle_size`: Number of close friends (0-15)
- `Post_frequency`: Social media post frequency (0-10)
- `Personality`: Target variable (Extrovert/Introvert)

### Container Configuration

The container is configured with:
- **Base Image**: Python 3.12 slim
- **Package Manager**: uv (for faster dependency installation)
- **Port**: 8501 (Streamlit default)
- **Working Directory**: /app
- **Entrypoint**: ./entrypoint.sh

### Troubleshooting

#### Container Won't Start
- Ensure Docker daemon is running
- Check if port 8501 is available
- Verify the image was built successfully

#### Application Errors
- Ensure `data/train.csv` exists and has the correct format
- Check container logs: `docker logs <container-name>`

#### Performance Issues
- The container includes system dependencies (gcc, g++) for numerical libraries
- Consider allocating more memory to Docker if needed