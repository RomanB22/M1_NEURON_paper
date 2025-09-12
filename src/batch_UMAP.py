from netpyne.batchtools.search import search
from pathlib import Path
import os
import pandas as pd

CWD = os.getcwd()

nameCluster = 'ssh_expanse_gpu'
directorySGE = 'M1_Manifolds_UMAP' #'ChannelopathiesGPU' M1_Manifolds M1_Manifolds_UMAP
directoryExpanse = 'M1_Manifolds_UMAP' #'ChannelopathiesGPU_Last' M1_Manifolds M1_Manifolds_UMAP
numSamples = 3000
PercentageChange = 0.2
minChg = (1-PercentageChange)
maxChg = (1+PercentageChange)

dataFrame = pd.read_csv('./manifolds/BaselineModels.csv') 
include = ['IIweights.0', 'IEweights.2', 'IIweights.2',
       'IIweights.1', 'EICellTypeGain.SOM', 'EICellTypeGain.PV',
       'EICellTypeGain.NGF', 'EICellTypeGain.VIP', 'weightLong.S1',
       'weightLong.S2', 'weightLong.TPO', 'weightLong.TVL', 'weightLong.OC',
       'EEGain', 'weightLong.cM1', 'weightLong.M2', 'IEweights.0',
       'IEweights.1']

chosenTrial = 0

row = dataFrame[include].iloc[chosenTrial]
params = {}

params = {
    col: [minChg * row[col], maxChg * row[col]]
    for col in include
}
# params['period'] = 'full_trial' # 'scaled_tone', 'scaled_prep', 'full_unlock' full_trial

# --- Define Constants and Common Settings ---

with open('ExpanseKey.txt') as f:
    SSH_KEY_PATH = f.readlines()[0]

# Common shell commands for setting up the Python environment
PYTHON_SETUP_CMDS = """
# Add project root and src to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH=$PYTHONPATH:$PWD/src
"""

CHECKPOINT = "./batchData/ray"
OUTPUT = "./batchData/optuna_batch"

IMPORTNEURONGPU ="""
export PATH=$HOME/neuronGPU/bin:$PATH
export PYTHONPATH=$HOME/neuronGPU/lib/python:$PYTHONPATH
"""

GPUCONFIG_DOWNSTATE = """
conda activate GPU
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib/":"/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib"
""" 

CONFIG_EXPANSE_CPU = """
source ~/.bashrc
module purge
module load shared
module load cpu/0.17.3b # most recent
module load slurm
module load sdsc
module load DefaultModules
module load openmpi/mlnx/gcc/64/4.1.5a1
conda activate NetPyNE
export LD_LIBRARY_PATH="/home/rbaravalle/miniconda/envs/NetPyNE/lib/python3.10/site-packages/mpi4py_mpich.libs/"
"""

CONFIG_EXPANSE_GPU = """
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
module load cmake/3.31.2/w4akk6u
"""


# --- Main Configuration Dictionary ---

config = {
    'sh_local': {
        'job_type': 'sh',
        'comm_type': 'socket',
        'host': '###',
        'key': '###',
        'remote_dir': CWD,
        'output_path': OUTPUT,
        'checkpoint_path': CHECKPOINT,
        'run_config': {
            'command': f"""
                unset DISPLAY
                conda activate M1_CEBRA
                {PYTHON_SETUP_CMDS}
                nrniv -python src/init_UMAP.py
            """
        }
    },
    'sge_gpu': {
        'job_type': 'sge',
        'comm_type': 'sfs',
        'host': '###',
        'key': '###',
        'remote_dir': '/ddn/rbarav/M1_Manifolds',
        'output_path': OUTPUT,
        'checkpoint_path': CHECKPOINT,
        'run_config': {
            'queue': 'gpu.q',
            'cores': 11,
            'vmem': '150G',
            'realtime': '15:00:00',
            'command': f"""
                {GPUCONFIG_DOWNSTATE}
                {IMPORTNEURONGPU}
                {PYTHON_SETUP_CMDS}
                mpiexec -n $NSLOTS ./x86_64/special -python -mpi src/init_UMAP.py
            """
        }
    },
    'sge_cpu': {
        'job_type': 'sge',
        'comm_type': 'sfs',
        'host': '###',
        'key': '###',
        'remote_dir': '/ddn/rbarav/M1_Manifolds',
        'output_path': './batchData/optuna_batch',
        'checkpoint_path': './batchData/ray',
        'run_config': {
            'queue': 'cpu.q',
            'cores': 50,
            'vmem': '200G',
            'realtime': '15:00:00',
            'command': f"""
                conda activate M1_dev
                export LD_LIBRARY_PATH="/ddn/rbarav/miniconda3/envs/M1_dev/lib/python3.10/site-packages/mpi4py_mpich.libs"
                {PYTHON_SETUP_CMDS}
                mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi src/init_UMAP.py
            """
        }
    },
    'ssh_sge_gpu': {
        'job_type': 'ssh_sge',
        'comm_type': 'sftp',
        'host': 'grid0',
        'key': '###',
        'remote_dir': '/ddn/rbarav/%s' % directorySGE,
        'output_path': './batchData/optuna_batch',
        'checkpoint_path': './batchData/ray_SGEGPU_3',
        'run_config': {
            'queue': 'gpu.q',
            'cores': 19,
            'vmem': '100G',
            'realtime': '15:00:00',
            'command': f"""
                {GPUCONFIG_DOWNSTATE}
                {IMPORTNEURONGPU}
                {PYTHON_SETUP_CMDS}
mpiexec -n $NSLOTS ./x86_64/special -python -mpi src/init_UMAP.py
            """
        }
    },
    'ssh_sge_cpu': {
        'job_type': 'ssh_sge',
        'comm_type': 'sftp',
        'host': 'grid0',
        'key': '###',
        'remote_dir': '/ddn/rbarav/%s' % directorySGE,
        'output_path': './batchData/optuna_batch',
        'checkpoint_path': './batchData/ray',
        'run_config': {
            'queue': 'cpu.q',
            'cores': 50,
            'vmem': '150G',
            'realtime': '15:00:00',
            'command': f"""
                conda activate M1_dev
                export LD_LIBRARY_PATH="/ddn/rbarav/miniconda3/envs/M1_dev/lib/python3.10/site-packages/mpi4py_mpich.libs"
                {PYTHON_SETUP_CMDS}
mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi src/init_UMAP.py
            """
        }
    },
    'ssh_expanse_cpu': {
        'job_type': 'ssh_slurm',
        'comm_type': 'sftp',
        'host': 'expanse0',
        'key': SSH_KEY_PATH,  # No key needed for this host
        'remote_dir': '/home/rbaravalle/%s' % directoryExpanse,
        'output_path': './batchData/optuna_batch',
        'checkpoint_path': "./batchData/ray_expanseCPU",
        'run_config': {
            'allocation': 'TG-MED240058',
            'realtime': '10:30:00',
            'nodes': 1,
            'coresPerNode': 96,
            'mem': '240G',
            'partition': 'compute',
            'custom': '',
            'email': 'romanbaravalle@gmail.com',
            'command': f"""
                {CONFIG_EXPANSE_CPU}
                {PYTHON_SETUP_CMDS}
time mpirun --bind-to none -n $SLURM_NTASKS ./x86_64/special -mpi -python src/init_UMAP.py
            """
        }
    },
    'ssh_expanse_gpu': {
        'job_type': 'ssh_slurm',
        'comm_type': 'sftp',
        'host': 'expanse0',
        'key': SSH_KEY_PATH,  # No key needed for this host
        'remote_dir': '/home/rbaravalle/%s' % directoryExpanse,
        'output_path': './batchData/optuna_batch',
        'checkpoint_path': "./batchData/ray_expanseUMAP",
        'run_config': {
            'allocation': 'TG-MED240058',
            'realtime': '10:30:00',
            'nodes': 1,
            'coresPerNode': 16,
            'mem': '128G',
            'partition': 'gpu-shared \n#SBATCH --gpus=3',
            'custom': '',
            'email': 'romanbaravalle@gmail.com',
            'command': f"""
                {CONFIG_EXPANSE_GPU}
                {IMPORTNEURONGPU}
                {PYTHON_SETUP_CMDS}
time mpirun --bind-to none -n $SLURM_NTASKS ./x86_64/special -mpi -python src/init_UMAP.py
            """
        }
    }
}

# --- Simplified Function Call ---

# Select the desired cluster configuration
# This variable would be set based on your execution environment
cluster_config = config[nameCluster]

# Unpack the cluster configuration directly into the function call using **
# This is much cleaner than manually specifying each argument
results = search(
    # --- Search-specific parameters ---
    label='optuna',
    params=params,          # Your search parameters
    metric='loss', # Use 2D full Wasserstein or "loss_sliced" for Sliced Wasserstein (faster-approximation fo the first one)
    mode='min',
    algorithm="optuna",
    max_concurrent=1,
    num_samples=numSamples,
    sample_interval=15,
    
    # --- Unpack all parameters from the chosen cluster config ---
    **cluster_config
)
