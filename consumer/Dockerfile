# Stage 1: Build dependencies
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime AS builder
FROM python:3.13-slim AS builder

# No prompts in install
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY consumer/ ./consumer
COPY src/ ./src

# Stage 2: Runtime
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM python:3.13-slim  


WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin


COPY --from=builder /app/consumer ./consumer
COPY --from=builder /app/src ./src

# Set working directory to consumer directly
WORKDIR /app/consumer

CMD ["python", "consumer.py"]