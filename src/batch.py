from netpyne.batchtools.search import search
from pathlib import Path
cwd = str(Path.cwd())

params = {'weightLong.TPO': [0.25, 0.75],
          'weightLong.S1': [0.25, 0.75],
          'weightLong.S2': [0.25, 0.75],
          'weightLong.cM1': [0.25, 0.75],
          'weightLong.M2': [0.25, 0.75],
          'weightLong.OC': [0.25, 0.75],
          'EEGain': [0.5, 1.5],
          'IEweights.0': [0.5, 1.5],
          'IEweights.1': [0.5, 1.5],
          'IEweights.2': [0.5, 1.5],
          'IIweights.0': [0.5, 1.5],
          'IIweights.1': [0.5, 1.5],
          'IIweights.2': [0.5, 1.5],
          }

# SGE GPU CONFIG
sge_config = {
    'queue': 'gpu.q',
    'cores': 19,
    'vmem': '90G', #90G
    'realtime': '15:00:00',
    'command': 'mpiexec -n $NSLOTS -hosts $(hostname) ./x86_64/special -python -mpi init.py'}

# use batch_shell_config if running directly on the machine
shell_config = {'command': 'nrniv -python src/init.py'}

# EXPANSE CONFIG
setup = """
source ~/.bashrc
source ~/default.sh
conda activate M1_batchTools
export LD_LIBRARY_PATH="/home/rbaravalle/.conda/envs/NetPyNE/lib/python3.10/site-packages/mpi4py_mpich.libs/"
"""
slurm_config = {
    'allocation': 'TG-MED240058',
    'realtime': '10:30:00',
    'nodes': 1,
    'coresPerNode': 96,
    'mem': '128G',
    'partition': 'compute',
    'email': 'romanbaravalle@gmail.com',
    'custom': setup,
    'command':'time mpirun -n 96 nrniv -python -mpi init.py'
}

# =======================
# job_type    , comm_type
# =======================
# 'sge'       , 'socket'    -> job submission through Sun Grid Engine, INET socket based communication
# 'sge'       , 'sfs'       -> job submission through Sun Grid Engine, communication via shared file system
# 'sge'       , None        -> job submission through Sun Grid Engine, no communication (only grid or random searches)
# 'ssh_sge'   , 'sftp'      -> remote SSH onto a gateway, job submission through Sun Grid Engine, communication via Secure FTP
# 'ssh_slurm' , 'sftp'      -> remote SSH onto a gateway, job submission through Slurm, communication via Secure FTP
# 'ssh_sge'   , None        -> remote SSH onto a gateway, job submission through Sun Grid Engine, no communication (only grid or random searches)
# 'ssh_slurm' , None        -> remote SSH onto a gateway, job submission through Slurm, no communication (only grid or random searches)
# 'sh'        , 'socket'    -> job run directly on local shell, INET socket based communication
# 'sh'        , 'sfs'       -> job run directly on local shell, communication via shared file system
# 'sh'        , None        -> job run directly on local shell, no communication (only grid or random searches)

results = search(job_type = 'sh', # or 'sh'
       comm_type = 'socket', # if a metric and mode is specified, some method of communicating with the host needs to be defined
       label = 'optuna',
       params = params,
       output_path = cwd+'/batchData/optuna_batch',
       checkpoint_path = cwd+'/batchData/ray',
       run_config = shell_config,
       metric = 'loss', # if a metric and mode is specified, the search will collect metric data and report on the optimal configuration
       mode = 'min',
       algorithm = "optuna",
       max_concurrent = 1,
    #    remote_dir='/home/rbaravalle/M1_CEBRA_BatchTools/src',
    #    host='expanse0',
    #    key='###',
       num_samples=2,
       )