conda activate M1
cd ChannelopathiesGPU
export PYTHONPATH=$PYTHONPATH:$PWD # do it in \src and in parent folder
cd src
export PYTHONPATH=$PYTHONPATH:$PWD # do it in \src and in parent folder
nrnivmodl -coreneuron ../mod/