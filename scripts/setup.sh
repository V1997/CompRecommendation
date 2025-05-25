#!/bin/bash

# Setup script for Real Estate AI Property Valuation System

echo "Setting up Real Estate AI Property Valuation System..."

# Check if Git LFS is available
if ! command -v git-lfs &> /dev/null; then
    echo "Git LFS not found. Please install Git LFS first:"
    echo "https://git-lfs.com/"
    exit 1
fi

# Initialize Git LFS and pull files
echo "Initializing Git LFS..."
git lfs install
git lfs pull

# Check if dataset file exists and has correct size
if [ -f "appraisals_dataset.json" ]; then
    size=$(wc -c < appraisals_dataset.json)
    size_mb=$((size / 1024 / 1024))
    echo "Dataset file size: ${size_mb}MB"
    if [ $size_mb -lt 20 ]; then
        echo "Warning: Dataset file seems too small. Try: git lfs pull"
    fi
else
    echo "Error: appraisals_dataset.json not found"
    echo "Make sure to run: git lfs pull"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install torch==2.0.1
pip install faiss-cpu==1.7.4
pip install shap==0.42.1
pip install lime==0.2.0.1
pip install geopy==2.3.0
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install pytest==7.4.0
pip install jupyter==1.0.0

# Create project structure
echo "Creating project structure..."
mkdir -p data/{raw,processed,models}
mkdir -p src/{data_preprocessing,feature_engineering,similarity_search,ranking,explainability,pipeline}
mkdir -p tests
mkdir -p notebooks
mkdir -p config
mkdir -p docs

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "2. Run: python data_loader.py"
echo "3. Start with notebooks/01_data_exploration.ipynb"