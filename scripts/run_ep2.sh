#!/bin/bash
#PBS -N torchtitan_ep2
#PBS -l walltime=00:30:00
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1
#PBS -l filesystems=flare:home

cd $PBS_O_WORKDIR

# Initialize modules
. /usr/share/lmod/lmod/init/bash
module load py-torch/2.9.0.dev20250804 py-ipex/xpu-main

export PYTHONPATH=$(pwd):$PYTHONPATH
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

PYTHON_EXE=/opt/aurora/25.190.0/spack/unified/0.10.1/install/linux-sles15-x86_64/gcc-13.3.0/python-venv-1.0-bdxpu45/bin/python3
MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec

# Use the FIRST node of the allocation as the master
MASTER_ADDR=$(hostname)
MASTER_PORT=29507

echo "Using Python: $PYTHON_EXE"
echo "Master Address: $MASTER_ADDR"

$MPIEXEC -n 2 -ppn 2 \
  -env MASTER_ADDR $MASTER_ADDR \
  -env MASTER_PORT $MASTER_PORT \
  -env CCL_PROCESS_LAUNCHER=pmix \
  -env FI_PROVIDER=psm3 \
  -env FI_PSM3_HALO=1 \
  -env CCL_BLOCKING_WAIT=1 \
  $PYTHON_EXE launch_wrapper.py torchtitan/train.py --job.config_file torchtitan/models/llama4/train_configs/llama4_ep2_triton_xpu.toml --training.steps=10
