#!/bin/bash
. /usr/share/lmod/lmod/init/bash
module purge
module load frameworks py-mpi4py/4.0.1

echo "Loaded frameworks + mpi4py."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); import mpi4py; print(f'mpi4py: {mpi4py.__version__}')"
