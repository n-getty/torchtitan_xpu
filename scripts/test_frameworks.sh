#!/bin/bash
. /usr/share/lmod/lmod/init/bash
module purge
module load frameworks

AURORA_INF=/home/ngetty/proj/vllm_gpt-oss/Aurora-Inferencing
export PYTHONPATH=$(pwd):$(pwd)/torchtitan:$AURORA_INF:$PYTHONPATH

echo "Loaded frameworks module."
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); import torchtitan; print('torchtitan imported'); from torchtitan_xpu import XPUExpertParallel; print('XPUExpertParallel imported')"
