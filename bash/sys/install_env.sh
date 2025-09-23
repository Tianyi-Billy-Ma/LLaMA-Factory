#!/bin/bash

# LLaMA-Factory Environment Installation Script
# This script successfully installs the LLaMA-Factory environment using uv
# Created: 2025-09-21

set -e  # Exit on any error

echo "🚀 Starting LLaMA-Factory environment installation..."

# Check if uv is installed, install if not found
if ! command -v uv &> /dev/null; then
    echo "📥 uv not found, installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Source the shell profile to make uv available in current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Verify uv installation
    if ! command -v uv &> /dev/null; then
        echo "❌ Error: Failed to install uv. Please install manually."
        echo "   Visit: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

echo "✅ uv found: $(uv --version)"

# Create virtual environment with Python 3.12 if .venv doesn't exist
if [ ! -d ".venv" ]; then
    echo "🐍 Creating virtual environment with Python 3.12..."
    uv venv --python 3.12
else
    echo "✅ Virtual environment (.venv) already exists"
fi

# Install dependencies from requirements.txt
echo "📦 Installing dependencies from requirements.txt..."
uv pip install -r requirements.txt

# Install the project in editable mode
echo "🔧 Installing LLaMA-Factory in editable mode..."
uv pip install -e .

# Verify installation
echo "🧪 Verifying installation..."
if source .venv/bin/activate && python -c "import llamafactory; print('✅ LLaMA-Factory successfully imported!')"; then
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📋 Usage Instructions:"
    echo "   1. Activate the virtual environment:"
    echo "      source .venv/bin/activate"
    echo ""
    echo "   2. Use Python normally:"
    echo "      python -c \"import llamafactory\""
    echo ""
    echo "   3. Use the CLI:"
    echo "      llamafactory-cli --help"
    echo ""
    echo "🔧 Available CLI commands:"
    echo "   • llamafactory-cli api      - Launch OpenAI-style API server"
    echo "   • llamafactory-cli chat     - Launch chat interface in CLI"
    echo "   • llamafactory-cli eval     - Evaluate models"
    echo "   • llamafactory-cli export   - Merge LoRA adapters and export model"
    echo "   • llamafactory-cli train    - Train models"
    echo "   • llamafactory-cli webchat  - Launch chat interface in Web UI"
    echo "   • llamafactory-cli webui    - Launch LlamaBoard"
    echo "   • llamafactory-cli version  - Show version info"
    echo ""
    echo "⚠️  Note: If 'uv run' fails due to dependency conflicts, use the activated venv instead."
else
    echo "❌ Installation verification failed!"
    exit 1
fi
