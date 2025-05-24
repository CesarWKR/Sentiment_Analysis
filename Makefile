# Variables
DOCKER_USER ?= $(shell echo $$DOCKER_USERNAME)
VERSION = latest
SLIM_TAG = slim

# Validate
ifndef DOCKER_USER
$(error ❌ DOCKER_USER is undefined. Please set it as an environment variable or in the Makefile)
endif

# Services with their own builds
SERVICES = producer consumer trainer

# Build all services
build:
    docker-compose build

# Start all services
up:
    docker-compose up -d

# Stop all services
down:
    docker-compose down

# Optimize images with docker-slim
slim:
    @for service in $(SERVICES); do \
        echo "⚡ Slimming service: $$service"; \
        docker-slim build --tag $$service:$(SLIM_TAG) $$service:$(VERSION); \
    done

# Tag slim images for DockerHub
tag:
    @for service in $(SERVICES); do \
        echo "🏷️ Tagging $$service"; \
        docker tag $$service:$(SLIM_TAG) $(DOCKER_USER)/$$service:$(SLIM_TAG); \
    done

# Push slim images to DockerHub
push:
    @for service in $(SERVICES); do \
        echo "🚀 Pushing $$service to DockerHub"; \
        docker push $(DOCKER_USER)/$$service:$(SLIM_TAG); \
    done

# Complete workflow: Build + Slim + Tag + Push
publish: build slim tag push
	@echo "✅ Optimized images successfully published."

# Clean up local images
clean:
    @for service in $(SERVICES); do \
        docker rmi $$service:$(VERSION) || true; \
        docker rmi $$service:$(SLIM_TAG) || true; \
        docker rmi $(DOCKER_USER)/$$service:$(SLIM_TAG) || true; \
    done