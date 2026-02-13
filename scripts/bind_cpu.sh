#!/bin/bash
# scripts/bind_cpu.sh

# Determine local rank from PMI vars (works for Cray PALS/MPICH)
RANK=${PMI_LOCAL_RANK:-${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-0}}}

# Ranks 0-5 -> Socket 0
# Ranks 6-11 -> Socket 1
# Assuming 12 ranks total, 2 sockets
SOCKET=$((RANK / 6))

# Execute the command with NUMA affinity
# -N <node>: bind to cpuset of node <node>
# -m <node>: bind to memory of node <node>
exec numactl -N $SOCKET -m $SOCKET "$@"
