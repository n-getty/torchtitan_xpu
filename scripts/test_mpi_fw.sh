#!/bin/bash
. /usr/share/lmod/lmod/init/bash
module purge
module load frameworks py-mpi4py

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
PYTHON_BIN=$(which python3)

echo "Testing mpiexec with python: $PYTHON_BIN"

# Check if we can just run hostname
$MPIEXEC -n 1 hostname

# Try running python with env dump
$MPIEXEC -n 1 --envall $PYTHON_BIN -c "import sys; print(f'Python: {sys.executable}'); import intel_extension_for_pytorch as ipex; print(f'IPEX: {ipex.__version__}')"
