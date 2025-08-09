#!/bin/bash
#SBATCH --job-name=NewTest
#SBATCH --account=TG-IBN140002 #MED240058 #IBN140002
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --gpus=3
#SBATCH --time=02:00:00
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err
#SBATCH --export=ALL

# >>> Conda setup
source ~/.bashrc 
conda activate NetPyNE_GPU
# <<< End Conda setup

# Load modules
echo "Loading modules..."
module purge
module use /cm/shared/apps/spack/0.21.2/gpu/dev/share/spack/lmod/linux-rocky8-x86_64/Core
module load nvhpc/24.11/2utxz5z
module load openmpi/mlnx/gcc/64/4.1.5a1
module load cmake/3.31.2/w4akk6u

cd /home/rbaravalle/M1_Manifolds

# Add PWD if not already in PYTHONPATH
if [[ ":$PYTHONPATH:" != *":$PWD:"* ]]; then
    export PYTHONPATH=$PYTHONPATH:$PWD
fi
# Add subfolder if not already in PYTHONPATH
SUBFOLDER="$PWD/src"
if [[ ":$PYTHONPATH:" != *":$SUBFOLDER:"* ]]; then
    export PYTHONPATH=$PYTHONPATH:$SUBFOLDER
fi

export PATH=$HOME/neuronGPU/bin:$PATH
export PYTHONPATH=$HOME/neuronGPU/lib/python:$PYTHONPATH

# export CORENRN_ENABLE_GPU=1
# export OMP_NUM_THREADS=1

# nrnivmodl -coreneuron mod/

# Use SLURM variable for automatic scaling
time mpirun -n $((SLURM_NTASKS-1)) ./x86_64/special -mpi -python src/init.py