#!/bin/bash
# Quick Deployment Script for FinalProject-Group5

set -e  # Exit on error

echo "=================================================================="
echo "        FinalProject-Group5 - Deployment Script                  "
echo "=================================================================="
echo ""

# Step 1: Clone/Update
echo "[1/6] Repository Setup"
if [ ! -d "FinalProject-Group5" ]; then
    echo "  - Cloning repository..."
    git clone https://github.com/mayur212626/FinalProject-Group5.git
fi
cd FinalProject-Group5
git fetch origin
git checkout mayur
git pull origin mayur
echo "  [DONE] Repository ready"
echo ""

# Step 2: Virtual Environment
echo "[2/6] Virtual Environment"
if [ ! -d "venv" ]; then
    echo "  - Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate
echo "  [DONE] Virtual environment activated"
echo ""

# Step 3: Dependencies
echo "[3/6] Installing Dependencies"
echo "  - Installing packages from requirements.txt..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo "  [DONE] Dependencies installed"
echo ""

# Step 4: Configuration
echo "[4/6] Configuration"
if [ ! -f ".env" ]; then
    echo "  - Creating .env from template..."
    cp .env.example .env
    echo "  WARNING: Please update .env with your API keys and preferences"
else
    echo "  [OK] .env file found"
fi
echo "  [DONE] Configuration ready"
echo ""

# Step 5: Verification
echo "[5/6] Verification"
echo "  - Verifying Python imports..."
python3 << 'PYEOF'
try:
    import torch
    import pandas as pd
    import numpy as np
    from models import PriceLSTMModel, FusionMLP
    from utils import compute_technical_indicators
    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}")
    exit(1)
PYEOF
echo "  [DONE] Verification passed"
echo ""

# Step 6: Project Structure
echo "[6/6] Project Structure"
echo "  [OK] Project ready at: $(pwd)"
echo ""
echo "  Directory structure:"
echo "    - models/         (PyTorch models)"
echo "    - utils/          (Utility functions)"
echo "    - scripts/        (Pipeline 01-07)"
echo "    - data/           (Data storage)"
echo "    - requirements.txt"
echo "    - DEPLOYMENT.md   (Setup guide)"
echo "    - Dockerfile      (Container config)"
echo ""

# Summary
echo "=================================================================="
echo "                 DEPLOYMENT SUCCESSFUL                            "
echo "=================================================================="
echo ""
echo "NEXT STEPS:"
echo "   1. Update .env with your configuration"
echo "   2. Run the pipeline:"
echo "      python scripts/01_download_data.py"
echo "      python scripts/02_feature_engineering.py"
echo "      ... (continue with steps 03-07)"
echo ""
echo "DOCUMENTATION:"
echo "   - DEPLOYMENT.md  - Complete deployment guide"
echo "   - README.md      - Project overview"
echo ""
echo "DOCKER DEPLOYMENT:"
echo "   docker-compose up -d"
echo ""
echo "For more information, see DEPLOYMENT.md"
