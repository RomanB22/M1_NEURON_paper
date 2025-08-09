unset DISPLAY
conda activate M1_CEBRA
export PYTHONPATH=$PYTHONPATH:$PWD # do it in \src and in parent folder
cd src
export PYTHONPATH=$PYTHONPATH:$PWD # do it in \src and in parent folder
cd ..
nrnivmodl -coreneuron mod/


# nohup command
nohup python -u src/batchNew.py > optunaGPU_SGE.txt & echo $! >> pids.pids

J4PXKKROVTM3R4ELLVAQ3CJCL6OUP2WN

nohup python -u src/batchNew.py > optunaGPU_Expanse.txt & echo $! >> pids.pids

nohup python -u src/batchNew.py > optunaCPU_Expanse.txt & echo $! >> pids.pids