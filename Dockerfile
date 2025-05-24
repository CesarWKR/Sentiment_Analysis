# =========================
# Stage 1: Build Stage
# =========================
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS builder


# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive 

# Create and define the working directory
WORKDIR /app

# Install system dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


# =========================
# Stage 2: Final Stage
# =========================

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive 

# Create and define the working directory
WORKDIR /app

# Copy the installed dependencies and application code from the builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY . .

# Command to run the application
CMD ["python", "main.py"]