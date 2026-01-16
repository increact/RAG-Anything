#!/bin/bash
# Quick start script for RAG-Anything Docker deployment

set -e

echo "🚀 Starting RAG-Anything API Server..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    if [ -f .env.docker ]; then
        cp .env.docker .env
        echo "✅ Created .env from .env.docker"
        echo "⚠️  Please edit .env and add your API keys before continuing"
        read -p "Press Enter to continue after editing .env..."
    else
        echo "❌ .env.docker template not found!"
        exit 1
    fi
fi

# Create necessary directories
mkdir -p output rag_storage logs

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Determine compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Build and start
echo "📦 Building Docker image..."
$COMPOSE_CMD build

echo "🚀 Starting services..."
$COMPOSE_CMD up -d

echo "⏳ Waiting for service to be ready..."
sleep 5

# Check health
echo "🏥 Checking health..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Service is healthy!"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ Service health check failed. Check logs with: $COMPOSE_CMD logs"
        exit 1
    fi
    echo "   Attempt $i/10..."
    sleep 3
done

echo ""
echo "✅ RAG-Anything API Server is running!"
echo ""
echo "📍 API Endpoint: http://localhost:8000"
echo "🏥 Health Check: http://localhost:8000/health"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "📋 Useful commands:"
echo "   View logs:    $COMPOSE_CMD logs -f rag-anything-api"
echo "   Stop:         $COMPOSE_CMD down"
echo "   Restart:      $COMPOSE_CMD restart rag-anything-api"
echo ""

