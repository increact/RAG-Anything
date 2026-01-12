#!/bin/bash
# Quick start script for RAG-Anything local development on macOS

set -e

echo "🍎 Starting RAG-Anything API Server (macOS Local Development)..."

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is optimized for macOS. You may want to use start.sh instead."
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from local template..."
    if [ -f env.local.example ]; then
        cp env.local.example .env
        echo "✅ Created .env from env.local.example"
        echo "⚠️  Please edit .env and add your API keys before continuing"
        read -p "Press Enter to continue after editing .env..."
    elif [ -f .env.docker ]; then
        cp .env.docker .env
        echo "✅ Created .env from .env.docker"
        echo "⚠️  Please edit .env and add your API keys before continuing"
        read -p "Press Enter to continue after editing .env..."
    else
        echo "❌ Environment template not found!"
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output rag_storage logs

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "   Please install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running."
    echo "   Please start Docker Desktop and try again."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed."
    exit 1
fi

# Determine compose command
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Check macOS architecture
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo "🍎 Detected Apple Silicon (M1/M2/M3) Mac"
    echo "   Note: GPU acceleration (MPS) may have limited support"
    echo "   Using CPU mode for better compatibility"
else
    echo "💻 Detected Intel Mac"
fi

# Build and start
echo ""
echo "📦 Building Docker image..."
$COMPOSE_CMD -f docker-compose.local.yml build

echo ""
echo "🚀 Starting services..."
$COMPOSE_CMD -f docker-compose.local.yml up -d

echo ""
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
        echo "❌ Service health check failed. Check logs with:"
        echo "   $COMPOSE_CMD -f docker-compose.local.yml logs"
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
echo "📚 Swagger Docs: http://localhost:8000/docs"
echo "📖 ReDoc: http://localhost:8000/redoc"
echo ""
echo "📋 Useful commands:"
echo "   View logs:    $COMPOSE_CMD -f docker-compose.local.yml logs -f rag-anything-api"
echo "   Stop:         $COMPOSE_CMD -f docker-compose.local.yml down"
echo "   Restart:      $COMPOSE_CMD -f docker-compose.local.yml restart rag-anything-api"
echo "   Rebuild:      $COMPOSE_CMD -f docker-compose.local.yml up -d --build"
echo ""
echo "💡 Tips for macOS:"
echo "   - Output files are in ./output (mounted from container)"
echo "   - Logs are in ./logs (mounted from container)"
echo "   - Models are cached in Docker volume (rag_models_cache)"
echo "   - For Apple Silicon, MPS support is experimental, CPU is recommended"
echo ""

