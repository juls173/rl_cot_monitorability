#!/bin/bash
set -e  # Exit on any error

echo "=========================================="
echo "Starting RunPod VERL Setup"
echo "=========================================="

REPO_URL="https://github.com/juls173/rl_cot_monitorability.git" 
WANDB_API_KEY=""

# ==========================================
# 1. Download and Install Conda
# ==========================================
echo "Step 1: Installing Conda..."
cd /workspace

wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda
rm miniconda.sh

# Initialize conda for this script session
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"

# Accept conda ToS for default channels
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Configure conda
conda config --system --prepend channels conda-forge
conda config --system --set auto_update_conda false

echo "✓ Conda installed successfully"

# ==========================================
# 2. Create and Activate Virtual Environment
# ==========================================
echo "Step 2: Creating conda environment..."
conda create -n verl python=3.10 -y

conda activate verl

# Verify activation
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

echo "✓ Virtual environment 'verl' created and activated"

# ==========================================
# 3. Clone VERL Repository
# ==========================================
echo "Step 3: Cloning VERL repository..."
cd /workspace

if [ -d "verl" ]; then
    echo "VERL directory already exists, pulling latest changes..."
    cd verl && git pull && cd /workspace
else
    git clone https://github.com/volcengine/verl.git
fi

echo "✓ VERL repository ready"

# ==========================================
# 4. Install Base Dependencies First
# ==========================================
echo "Step 4: Installing base dependencies..."

# Install essential packages first
pip install --upgrade pip setuptools wheel
pip install datasets huggingface_hub wandb

echo "✓ Base dependencies installed"

# ==========================================
# 5. Install VERL Dependencies
# ==========================================
echo "Step 5: Installing VERL dependencies (this may take a while)..."
cd /workspace/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

echo "✓ VERL dependencies installed"

# ==========================================
# 6. Install VERL in Editable Mode
# ==========================================
echo "Step 6: Installing VERL in editable mode..."
cd /workspace/verl
pip install --no-deps -e .

echo "✓ VERL installed in editable mode"

# ==========================================
# 7. Clone Code Repository
# ==========================================
echo "Step 7: Cloning your repository..."
cd /workspace
REPO_NAME=$(basename ${REPO_URL} .git)

if [ -d "${REPO_NAME}" ]; then
    echo "Repository '${REPO_NAME}' already exists, pulling latest changes..."
    cd ${REPO_NAME} && git pull && cd /workspace
else
    git clone ${REPO_URL}
fi

echo "✓ Code repository cloned"

# ==========================================
# 8. Download GSM8K Dataset
# ==========================================
echo "Step 8: Downloading GSM8K dataset..."
cd /workspace/verl/examples/data_preprocess

# Create data directory if it doesn't exist
mkdir -p /workspace/data/gsm8k

# Run the preprocessing script
python3 gsm8k.py --local_save_dir /workspace/data/gsm8k

echo "✓ GSM8K dataset downloaded to /workspace/data/gsm8k"

# ==========================================
# 9. Configure W&B Login
# ==========================================
echo "Step 9: Configuring Weights & Biases..."

if [ -n "${WANDB_API_KEY}" ]; then
    export WANDB_API_KEY="${WANDB_API_KEY}"
    wandb login ${WANDB_API_KEY}
    echo "✓ W&B configured"
else
    echo "⚠ Warning: WANDB_API_KEY is empty. Skipping W&B login."
fi

# ==========================================
# 10. Make Your Script Executable
# ==========================================
echo "Step 10: Setting up your training script..."
cd /workspace

if [ -f "${REPO_NAME}/scripts/run_grpo_LoRA.sh" ]; then
    chmod +x ${REPO_NAME}/scripts/run_grpo_LoRA.sh
    echo "✓ run_grpo_LoRA.sh script is now executable"
else
    echo "⚠ Warning: run_grpo_LoRA.sh script not found in ${REPO_NAME}/scripts/"
fi

# ==========================================
# 11. Setup Shell Configuration for Future Sessions
# ==========================================
echo "Step 11: Configuring shell for future sessions..."

# Add conda initialization to bashrc
cat >> ~/.bashrc << 'EOF'

# >>> conda initialize >>>
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
conda activate verl
# <<< conda initialize <<<
EOF

# Add WANDB_API_KEY if provided
if [ -n "${WANDB_API_KEY}" ]; then
    echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> ~/.bashrc
fi

echo "✓ Shell configuration updated"

# ==========================================
# Final Setup
# ==========================================
cd /workspace

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "Environment: verl"
echo "VERL location: /workspace/verl"
echo "Your repo location: /workspace/${REPO_NAME}"
echo "Dataset location: /workspace/data/gsm8k"
echo ""
echo "To run your training script:"
echo "  cd /workspace/${REPO_NAME}/scripts"
echo "  ./run_grpo_LoRA.sh"
echo ""
echo "For new terminal sessions, conda will auto-activate 'verl'"
echo "=========================================="
