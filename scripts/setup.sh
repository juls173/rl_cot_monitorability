#!/bin/bash
set -e  # Exit on any error

echo "=========================================="
echo "Starting RunPod VERL Setup"
echo "=========================================="

REPO_URL="https://github.com/juls173/rl_cot_monitorability.git" 
WANDB_API_KEY="your_wandb_api_key_here"

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

echo "✓ Conda installed successfully"

# ==========================================
# 2. Create Virtual Environment
# ==========================================
echo "Step 2: Creating conda environment..."
conda create -n verl python=3.10 -y
conda activate verl

# Ensure conda hook is available for future sessions
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"
echo 'source /workspace/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc
echo 'conda activate verl' >> ~/.bashrc

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
# 5. Install VERL Dependencies
# ==========================================
echo "Step 5: Installing VERL dependencies (this may take a while)..."
cd /workspace/verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

echo "✓ VERL dependencies installed"

# ==========================================
# 4. Install VERL in Editable Mode (no deps first)
# ==========================================
echo "Step 4: Installing VERL in editable mode (no deps)..."
cd /workspace/verl
pip install --no-deps -e .

echo "✓ VERL installed in editable mode"

# ==========================================
# 6. Clone Code Repository
# ==========================================
echo "Step 6: Cloning your repository..."
cd /workspace
REPO_NAME=$(basename ${REPO_URL} .git)

if [ -d "${REPO_NAME}" ]; then
    echo "Repository '${REPO_NAME}' already exists, pulling latest changes..."
    cd ${REPO_NAME} && git pull && cd /workspace
else
    git clone ${REPO_URL}
fi

echo "✓ Your repository ready"

# ==========================================
# 7. Download GSM8K Dataset
# ==========================================
echo "Step 7: Downloading GSM8K dataset..."
cd /workspace/verl/examples/data_preprocess

# Create data directory if it doesn't exist
mkdir -p ~/data/gsm8k

# Run the preprocessing script
python3 gsm8k.py --local_save_dir ~/data/gsm8k

echo "✓ GSM8K dataset downloaded to ~/data/gsm8k"

# ==========================================
# 8. Configure W&B Login
# ==========================================
echo "Step 8: Configuring Weights & Biases..."

# Set WANDB_API_KEY as environment variable
export WANDB_API_KEY="${WANDB_API_KEY}"
echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> ~/.bashrc

# Login to wandb
wandb login ${WANDB_API_KEY}

echo "✓ W&B configured"

# ==========================================
# 9. Make Your Script Executable
# ==========================================
echo "Step 9: Setting up your training script..."
cd /workspace

if [ -f "${REPO_NAME}/run_grpo_LoRA" ]; then
    chmod +x ${REPO_NAME}/run_grpo_LoRA
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
