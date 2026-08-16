#!/bin/bash

# 3GPP RAG Assistant - Quick Setup Script
# This script automates the initial setup process

set -e  # Exit on error

echo "🚀 3GPP RAG Assistant - Setup Script"
echo "===================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3,9)' 2>/dev/null; then
    echo "❌ Error: Python 3.9 or higher is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "   ✅ pip upgraded"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
echo "   ✅ All dependencies installed"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "🔐 Creating .env file..."
    cp .env.example .env
    echo "   ✅ .env file created"
    echo "   ⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
else
    echo "⚠️  .env file already exists, skipping"
fi

# Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/raw data/processed data/vectordb
echo "   ✅ Data directories created"

# Check if OpenAI API key is set
echo ""
if grep -q "your_openai_api_key_here" .env 2>/dev/null; then
    echo "⚠️  WARNING: OpenAI API key not configured!"
    echo "   Please edit .env and add your API key:"
    echo "   nano .env"
else
    echo "✅ OpenAI API key appears to be configured"
fi

# Final instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Add your OpenAI API key to .env (if not done)"
echo "   2. Download 3GPP specs to data/raw/"
echo "   3. Run: python src/core/document_processor.py"
echo "   4. Start API: uvicorn src.api.main:app --reload"
echo ""
echo "📖 For detailed instructions, see docs/GETTING_STARTED.md"
echo ""
echo "🎉 Happy coding!"
