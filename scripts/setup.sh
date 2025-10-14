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

# Source conda
source /workspace/miniconda/etc/profile.d/conda.sh
export PATH="/workspace/miniconda/bin:$PATH"

# Accept conda ToS for default channels
/workspace/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
/workspace/miniconda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Configure conda
/workspace/miniconda/bin/conda config --system --prepend channels conda-forge
/workspace/miniconda/bin/conda config --system --set auto_update_conda false

echo "✓ Conda installed successfully"

# ==========================================
# 2. Create Virtual Environment
# ==========================================
echo "Step 2: Creating conda environment..."
/workspace/miniconda/bin/conda create -n verl python=3.10 -y

# Initialize conda for bash
/workspace/miniconda/bin/conda init bash

# Source the bash profile to load conda functions
source ~/.bashrc

# Activate the environment
conda activate verl

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
# 4. Install VERL Dependencies
# ==========================================
echo "Step 4: Installing VERL dependencies (this may take a while)..."
cd /workspace/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

echo "✓ VERL dependencies installed"

# ==========================================
# 5. Install Additional Required Packages
# ==========================================
echo "Step 5: Installing additional required packages..."
pip install datasets huggingface_hub

echo "✓ Additional packages installed"

# ==========================================
# 6. Install VERL in Editable Mode (no deps first)
# ==========================================
echo "Step 6: Installing VERL in editable mode (no deps)..."
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

echo "✓ code repository cloned"

# ==========================================
# 8. Download GSM8K Dataset
# ==========================================
echo "Step 8: Downloading GSM8K dataset..."
cd /workspace/verl/examples/data_preprocess

# Create data directory if it doesn't exist
mkdir -p ~/../workspace/data/gsm8k

# Run the preprocessing script
python3 gsm8k.py --local_save_dir ~/../workspace/data/gsm8k

echo "✓ GSM8K dataset downloaded to ~/data/gsm8k"

# ==========================================
# 9. Configure W&B Login
# ==========================================
echo "Step 9: Configuring Weights & Biases..."

# Set WANDB_API_KEY as environment variable
export WANDB_API_KEY="${WANDB_API_KEY}"
echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> ~/.bashrc

# Login to wandb (only if API key is provided)
if [ -n "${WANDB_API_KEY}" ]; then
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

if [ -f "${REPO_NAME}/scripts/run_grpo_LoRA" ]; then
    chmod +x ${REPO_NAME}/scripts/run_grpo_LoRA.sh
    echo "✓ run_grpo_LoRA script is now executable"
else
    echo "⚠ Warning: run_grpo_LoRA script not found in ${REPO_NAME}/"
fi

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
echo "Dataset location: ~/data/gsm8k"
echo ""
echo "To run your training script:"
echo "  cd /workspace"
echo "  ./${REPO_NAME}/run_grpo_LoRA"
echo ""
echo "Environment is already activated!"
echo "=========================================="
