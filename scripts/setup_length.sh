#!/bin/bash
set -e  # Exit on any error

echo "=========================================="
echo "Starting RunPod VERL Setup"
echo "=========================================="

REPO_URL="https://github.com/juls173/rl_cot_monitorability.git" 
REPO_BRANCH="baram"
WANDB_API_KEY="6dff329b191825f14c13f6a4600ec43b34a68baf"
NUM_BUDGET_COPIES=2
BUDGET_VALUES="50 500"

# ==========================================
# 1. Download and Install Conda
# ==========================================
echo "Step 1: Installing Conda..."
cd /workspace

if [ -d "/workspace/miniconda" ]; then
    echo "Conda already installed, skipping installation..."
else
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /workspace/miniconda
    rm miniconda.sh
fi

# Source conda
source /workspace/miniconda/etc/profile.d/conda.sh

echo "✓ Conda installed successfully"

# Accept Conda Terms of Service
echo "Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ==========================================
# 2. Create Virtual Environment
# ==========================================
echo "Step 2: Creating conda environment..."

# Check if environment already exists
if conda env list | grep -q "^verl "; then
    echo "Environment 'verl' already exists, skipping creation..."
else
    conda create -n verl python=3.10 -y
fi

# Ensure conda hook is available for future sessions
eval "$(/workspace/miniconda/bin/conda shell.bash hook)"

# Add to bashrc only if not already present
if ! grep -q "conda activate verl" ~/.bashrc; then
    echo 'source /workspace/miniconda/etc/profile.d/conda.sh' >> ~/.bashrc
    echo 'conda activate verl' >> ~/.bashrc
fi

conda activate verl

# Double-check that verl environment has been activated 
if [ "$CONDA_DEFAULT_ENV" != "verl" ]; then
    echo "Error: verl environment not activated; got ${CONDA_DEFAULT_ENV:-none}"
    exit 1
fi

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

# Check if dependencies are already installed by looking for a marker file
if [ -f "/workspace/verl/.deps_installed" ]; then
    echo "VERL dependencies already installed, skipping..."
else
    USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
    touch /workspace/verl/.deps_installed
fi

echo "✓ VERL dependencies installed"

# ==========================================
# 5. Install VERL in Editable Mode (no deps first)
# ==========================================
echo "Step 5: Installing VERL in editable mode (no deps)..."
cd /workspace/verl

# Check if VERL is already installed
if pip show verl > /dev/null 2>&1; then
    echo "VERL already installed in editable mode, skipping..."
else
    pip install --no-deps -e .
fi

echo "✓ VERL installed in editable mode"

# ==========================================
# 6. Clone Code Repository
# ==========================================
echo "Step 6: Cloning your repository..."
cd /workspace
REPO_NAME=$(basename ${REPO_URL} .git)

if [ -d "${REPO_NAME}" ]; then
    echo "Repository '${REPO_NAME}' already exists, pulling latest changes..."
    cd ${REPO_NAME} && git checkout ${REPO_BRANCH} && git pull && cd /workspace
else
    git clone -b ${REPO_BRANCH} ${REPO_URL}
fi

echo "✓ code repository cloned"

# ==========================================
# 7. Download GSM8K Dataset
# ==========================================
echo "Step 7: Downloading GSM8K dataset..."
cd /workspace/verl/examples/data_preprocess

# Create data directory if it doesn't exist
mkdir -p ~/../workspace/data/gsm8k

# Check if dataset already exists
DATA_DIR=~/../workspace/data/gsm8k_${NUM_BUDGET_COPIES}
if [ -f "${DATA_DIR}/train.parquet" ] && [ -f "${DATA_DIR}/test.parquet" ]; then
    echo "GSM8K dataset already exists, skipping download..."
else
    # Run the preprocessing script
    python3 /workspace/${REPO_NAME}/scripts/gsm8k_token_budget.py --local_save_dir ${DATA_DIR} --num_budget_copies ${NUM_BUDGET_COPIES} --budget_values ${BUDGET_VALUES}
fi

echo "✓ GSM8K dataset downloaded to ~/data/gsm8k_${NUM_BUDGET_COPIES}"

# ==========================================
# 8. Configure W&B Login
# ==========================================
echo "Step 8: Configuring Weights & Biases..."

# Set WANDB_API_KEY as environment variable
export WANDB_API_KEY="${WANDB_API_KEY}"

# Add to bashrc only if not already present
if ! grep -q "WANDB_API_KEY" ~/.bashrc; then
    echo "export WANDB_API_KEY=${WANDB_API_KEY}" >> ~/.bashrc
fi

# Login to wandb
wandb login ${WANDB_API_KEY}

echo "✓ W&B configured"

# ==========================================
# 9. Make Your Script Executable
# ==========================================
echo "Step 9: Setting up your training script..."
cd /workspace

if [ -f "${REPO_NAME}/scripts/run_grpo_LoRA_length.sh" ]; then
    chmod +x ${REPO_NAME}/scripts/run_grpo_LoRA_length.sh
    echo "✓ run_grpo_LoRA_length script is now executable"
else
    echo "⚠ Warning: run_grpo_LoRA_length script not found in ${REPO_NAME}/scripts/"
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
echo "Dataset location: ~/data/gsm8k_${NUM_BUDGET_COPIES}"
echo ""
echo "To run your training script:"
echo "  cd /workspace"
echo "  ./${REPO_NAME}/scripts/run_grpo_LoRA_length"
echo ""
echo "⚠ Note: To activate the environment in your current shell, run:"
echo "  source ~/.bashrc"
echo "  # OR start a new terminal session"
echo "=========================================="
