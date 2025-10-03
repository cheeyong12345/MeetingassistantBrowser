#!/bin/bash
# Setup script for Meeting Assistant Browser Version

set -e

echo "=========================================="
echo "Meeting Assistant - Browser Version Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Found Python $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/meetings
mkdir -p data/temp
mkdir -p models

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Download STT models (if not already present):"
echo "   - Whisper models: See README_BROWSER.md"
echo "   - Vosk models: https://alphacephei.com/vosk/models"
echo ""
echo "3. Configure settings (optional):"
echo "   Edit config.yaml"
echo ""
echo "4. Run the application:"
echo "   python web_app_browser.py"
echo ""
echo "5. Open browser:"
echo "   http://localhost:8000"
echo ""
echo "=========================================="
