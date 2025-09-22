#!/bin/bash

# LLaMA-Factory Environment Activation Helper
# This script helps activate and use the installed environment
# Created: 2025-09-21

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run ./bash/install_env.sh first"
    exit 1
fi

echo "🚀 Activating virtual environment..."
source .venv/bin/activate
echo "✅ Environment activated!"

