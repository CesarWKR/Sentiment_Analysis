FROM python:3.13-slim AS builder

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the trained model and additional dependencies for FastAPI
COPY ../Roberta_sentiment_model/ ./Roberta_sentiment_model
COPY . .  

# Run the FastAPI application
CMD ["uvicorn", "inference_api:app", "--host", "0.0.0.0", "--port", "8000"]
