#!/bin/bash
. /usr/share/lmod/lmod/init/bash
module purge
module load frameworks py-mpi4py

MPIEXEC=/opt/cray/pals/1.8/bin/mpiexec
PYTHON_BIN=python3

echo "Testing mpiexec python path..."
$MPIEXEC -n 1 --envall $PYTHON_BIN -c "import sys; print(f'Path: {sys.path}'); import intel_extension_for_pytorch as ipex; print(f'IPEX: {ipex.__file__}')"
