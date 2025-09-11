#!/bin/bash
#SBATCH --job-name=neuron_mod_gpu
#SBATCH --account=TG-IBN140002
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gpus=3
#SBATCH --time=02:00:00
#SBATCH --output=neuron_mod_%j.out
#SBATCH --error=neuron_mod_%j.err

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Load modules
# >>> Conda setup
source ~/.bashrc 
module purge
conda activate NetPyNE_GPU
# <<< End Conda setup

# Load modules
echo "Loading modules..."
module use /cm/shared/apps/spack/0.21.2/gpu/dev/share/spack/lmod/linux-rocky8-x86_64/Core
module load nvhpc/24.11/2utxz5z
module load openmpi/mlnx/gcc/64/4.1.5a1
export PATH=$HOME/neuronGPU/bin:$PATH
export PYTHONPATH=$HOME/neuronGPU/lib/python:$PYTHONPATH
# Add project root and src to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH=$PYTHONPATH:$PWD/src

echo "Loaded modules:"
module list

echo "=========================================="
echo "Environment and Tool Check"
echo "=========================================="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NVCC available:"
which nvcc
nvcc --version

echo "NRNIVMODL available:"
which nrnivmodl
nrnivmodl --help | head -10

echo "=========================================="
echo "MOD File Compilation"
echo "=========================================="

# Check if MOD files exist
if [ -d "mod" ] || [ -n "$(find . -name "*.mod" -maxdepth 2)" ]; then
    echo "Found MOD files, compiling for GPU coreNEURON..."
    
    # Method 1: If MOD files are in a 'mod_files' directory
    if [ -d "mod" ]; then
        echo "Compiling MOD files from mod/ directory..."
        nrnivmodl -coreneuron mod/
    
    # Method 2: If MOD files are in current directory
    elif [ -n "$(find . -name "*.mod" -maxdepth 1)" ]; then
        echo "Compiling MOD files from current directory..."
        nrnivmodl -coreneuron .
    
    # Method 3: If MOD files are in subdirectories
    else
        echo "Compiling all MOD files found..."
        nrnivmodl -coreneuron $(find . -name "*.mod" -maxdepth 2 -exec dirname {} \; | sort -u | head -1)
    fi
    
    # Check compilation results
    if [ $? -eq 0 ]; then
        echo "MOD compilation successful!"
        echo "Generated files:"
        ls -la x86_64/ 2>/dev/null || ls -la ./special 2>/dev/null || echo "No compilation output directory found"
        
        # Check for coreNEURON specific files
        if [ -d "x86_64" ]; then
            echo "coreNEURON files:"
            find x86_64 -name "*coreneuron*" -o -name "*.so" | head -10
        fi
    else
        echo "MOD compilation failed!"
        echo "This might be due to:"
        echo "1. Missing CUDA compiler (nvcc)"
        echo "2. Incompatible MOD files"
        echo "3. Missing dependencies"
        exit 1
    fi
    
else
    echo "No MOD files found - compiling default mechanisms only..."
    nrnivmodl -coreneuron
    
    if [ $? -eq 0 ]; then
        echo "Default mechanism compilation successful!"
    else
        echo "Default mechanism compilation failed!"
        exit 1
    fi
fi

echo "=========================================="
echo "Python Path Setup"
echo "=========================================="

# Add compiled mechanisms to Python path
if [ -d "x86_64" ]; then
    export PYTHONPATH="$(pwd)/x86_64:$PYTHONPATH"
    echo "Added $(pwd)/x86_64 to PYTHONPATH"
fi

# Set NEURON library path
if [ -d "x86_64" ]; then
    export NRNHOME="$(pwd)/x86_64"
    echo "Set NRNHOME to $(pwd)/x86_64"
fi

echo "Current PYTHONPATH: $PYTHONPATH"

echo "=========================================="
echo "Running NEURON Simulation"
echo "=========================================="

# Test the compiled mechanisms
python << 'EOF'
import neuron
from neuron import h
import sys
import os

print("Testing compiled mechanisms...")

# Load the compiled mechanisms
try:
    if os.path.exists('./x86_64/libnrnmech.so'):
        h.nrn_load_dll('./x86_64/libnrnmech.so')
        print("Successfully loaded compiled mechanisms!")
    else:
        print("No compiled mechanisms found, using default...")
except Exception as e:
    print(f"Error loading mechanisms: {e}")

# Test coreNEURON
try:
    from neuron import coreneuron
    coreneuron.enable = True
    coreneuron.gpu = True
    print("coreNEURON with GPU enabled successfully!")
except Exception as e:
    print(f"coreNEURON setup error: {e}")

# List available mechanisms
print("\nAvailable mechanisms:")
mech_list = []
for i in range(int(h.nrn_mech_count())):
    mech_list.append(h.nrn_mech_name(i))
    
for mech in sorted(mech_list):
    print(f"  {mech}")

print(f"\nTotal mechanisms available: {len(mech_list)}")
EOF

echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="