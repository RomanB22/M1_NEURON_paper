from netpyne.batchtools.search import search
from pathlib import Path
cwd = str(Path.cwd())

params = {'weightLong.TPO': [0.25, 0.75],
          'weightLong.TVL': [0.25, 0.75],
          'weightLong.S1': [0.25, 0.75],
          'weightLong.S2': [0.25, 0.75],
          'weightLong.cM1': [0.25, 0.75],
          'weightLong.M2': [0.25, 0.75],
          'weightLong.OC': [0.25, 0.75],
          'EEGain': [0.5, 1.5],
          'IEweights.0': [0.5, 1.5],    ## L2/3+4
          'IEweights.1': [0.5, 1.5],    ## L5
          'IEweights.2': [0.5, 1.5],    ## L6
          'IIweights.0': [0.5, 1.5],    ## L2/3+4
          'IIweights.1': [0.5, 1.5],    ## L5
          'IIweights.2': [0.5, 1.5],    ## L6
        #   'EICellTypeGain.PV': [0.1, 4.0],    
        #   'EICellTypeGain.SOM': [0.1, 4.0],    
        #   'EICellTypeGain.VIP': [0.1, 4.0],    
        #   'EICellTypeGain.NGF': [0.1, 4.0],   
          }

nameCluster = 'ssh_sge_gpu'

config = {
    'sh_local': {'job_type': 'sh',
                 'comm_type': 'socket',
                 'output_path': cwd+'/batchData/optuna_batch',
                 'checkpoint_path': cwd+'/batchData/ray',
                 'host': '###',
                 'remote_dir': cwd,
                 'key': '###',  # replace with your SSH key                
                 'run_config': {'command': ('unset DISPLAY \n'
                                            'conda activate M1_CEBRA \n'                     
                                            'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                            'cd src \n'
                                            'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                            'cd .. \n'
                                            'nrniv -python src/init.py')}
                 },
    'sge_gpu': {'job_type': 'sge',
                    'comm_type': 'sfs',
                    'host': '###',
                    'remote_dir': '/ddn/rbarav/M1_Manifolds',
                    'key': '###',  # replace with your SSH key
                    'output_path':'./batchData/optuna_batch',
                    'checkpoint_path': './batchData/ray',
                    'run_config':  {'queue': 'gpu.q',
                                    'cores': 10,
                                    'vmem': '150G',
                                    'realtime': '15:00:00',
                                    'command': ('conda activate GPU  \n'
                                                'export PATH=$HOME/neuronGPU/bin:$PATH \n' 
                                                'export PYTHONPATH=$HOME/neuronGPU/lib/python:$PYTHONPATH \n'
                                                'export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib/":"/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib" \n'  
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd src \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd .. \n'                                               
                                                'mpiexec -n $NSLOTS ./x86_64/special -python -mpi src/init.py')}
    },
    'sge_cpu': { 'job_type': 'sge',
                    'comm_type': 'sfs',
                    'host': '###',
                    'remote_dir': '/ddn/rbarav/M1_Manifolds',
                    'key': '###',  # replace with your SSH key
                    'output_path':'./batchData/optuna_batch',
                    'checkpoint_path': './batchData/ray',
                    'run_config':  {'queue': 'cpu.q',
                                    'cores': 50,
                                    'vmem': '200G',
                                    'realtime': '15:00:00',
                                    'command': ('conda activate M1_dev  \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd src \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd .. \n'
                                                'export LD_LIBRARY_PATH="/ddn/rbarav/miniconda3/envs/M1_dev/lib/python3.10/site-packages/mpi4py_mpich.libs" \n'    
                                                'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi src/init.py')}
    },
    'ssh_sge_gpu': {'job_type': 'ssh_sge',
                    'comm_type': 'sftp',
                    'host': 'grid0',
                    'remote_dir': '/ddn/rbarav/M1_Manifolds',
                    'key': '###',  # replace with your SSH key
                    'output_path':'./batchData/optuna_batch',
                    'checkpoint_path': './batchData/ray',
                    'run_config':  {'queue': 'gpu.q',
                                    'cores': 10, 
                                    'vmem': '100G',
                                    'realtime': '15:00:00',
                                    'command': ('conda activate GPU  \n'
                                                'export PATH=$HOME/neuronGPU/bin:$PATH \n' 
                                                'export PYTHONPATH=$HOME/neuronGPU/lib/python:$PYTHONPATH \n'
                                                'export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib/":"/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/compilers/lib" \n'  
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd src \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd .. \n'                                               
                                                'mpiexec -n $NSLOTS ./x86_64/special -python -mpi src/init.py')
                                                # Put half of the cores you want to use
                                                # 'mpiexec -n 10 -x CUDA_VISIBLE_DEVICES=0 ./x86_64/special -python -mpi src/init.py : -n 10 -x CUDA_VISIBLE_DEVICES=1 ./x86_64/special -python -mpi src/init.py')
                                                }
    },
    'ssh_sge_cpu': { 'job_type': 'ssh_sge',
                    'comm_type': 'sftp',
                    'host': 'grid0',
                    'remote_dir': '/ddn/rbarav/M1_Manifolds',
                    'key': '###',  # replace with your SSH key
                    'output_path':'./batchData/optuna_batch',
                    'checkpoint_path': './batchData/ray',
                    'run_config':  {'queue': 'cpu.q',
                                    'cores': 50,
                                    'vmem': '150G',
                                    'realtime': '15:00:00',
                                    'command': ('conda activate M1_dev \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd src \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd .. \n'
                                                'export LD_LIBRARY_PATH="/ddn/rbarav/miniconda3/envs/M1_dev/lib/python3.10/site-packages/mpi4py_mpich.libs" \n'    
                                                'mpiexec -n $NSLOTS -hosts $(hostname) nrniv -python -mpi src/init.py')}
    },
    'ssh_expanse_cpu': { 'job_type': 'ssh_slurm',
                    'comm_type': 'sftp',
                    'host': 'expanse0',
                    'remote_dir': '/home/rbaravalle/M1_Manifolds',
                    'key': '###',  # replace with your SSH key
                    'output_path': '/home/rbaravalle/M1_Manifolds/batchData/optuna_batch',
                    'checkpoint_path': cwd+'/batchData/ray',
                    'run_config':  {'allocation': 'TG-MED240058',
                                    'realtime': '10:30:00',
                                    'nodes': 1,
                                    'coresPerNode': 96,
                                    'mem': '128G',
                                    'partition': 'compute',
                                    'email': 'romanbaravalle@gmail.com',
                                    'custom': ('source ~/.bashrc \n'
                                                'source ~/default.sh\n'
                                                'conda activate M1_batchTools\n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd src \n'
                                                'export PYTHONPATH=$PYTHONPATH:$PWD \n' # do it in \src and in parent folder
                                                'cd .. \n'
                                                'export LD_LIBRARY_PATH="/home/rbaravalle/.conda/envs/NetPyNE/lib/python3.10/site-packages/mpi4py_mpich.libs/" \n'),
                                    'command': 'time mpirun -n 96 nrniv -python -mpi src/init.py'}
    }
}

results = search(job_type = config[nameCluster]['job_type'], # job_type defines how the job is submitted to the cluster, e.g. 'ssh_sge', 'ssh_slurm', 'sge', 'sh'
       comm_type = config[nameCluster]['comm_type'], # if a metric and mode is specified, some method of communicating with the host needs to be defined
       label = 'optuna',
       params = params,
       output_path = config[nameCluster]['output_path'],
       checkpoint_path = config[nameCluster]['checkpoint_path'],
       run_config = config[nameCluster]['run_config'],
       metric = 'loss', # if a metric and mode is specified, the search will collect metric data and report on the optimal configuration
       mode = 'min',
       algorithm = "optuna",
       max_concurrent = 1,
       remote_dir=config[nameCluster]['remote_dir'],
       host=config[nameCluster]['host'],
       key=config[nameCluster]['key'],
       num_samples=200,
       sample_interval=15
       )

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