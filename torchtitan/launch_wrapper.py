import os
import subprocess
import sys

# Map PALS/MPI variables to PyTorch variables
os.environ["RANK"] = os.environ.get("PALS_RANKID", os.environ.get("RANK", "0"))
os.environ["WORLD_SIZE"] = os.environ.get("PALS_SIZE", os.environ.get("PALS_LOCAL_SIZE", os.environ.get("WORLD_SIZE", "1")))
os.environ["LOCAL_RANK"] = os.environ.get("PALS_LOCAL_RANKID", os.environ.get("LOCAL_RANK", "0"))

# Default Master Address to the first allocated node if not set
if "MASTER_ADDR" not in os.environ:
    os.environ["MASTER_ADDR"] = "x4300c2s0b0n0"
if "MASTER_PORT" not in os.environ:
    os.environ["MASTER_PORT"] = "29511"

# Print for debugging
if os.environ["LOCAL_RANK"] == "0":
    print(f"RANK: {os.environ['RANK']}, WORLD_SIZE: {os.environ['WORLD_SIZE']}, MASTER: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    # print some env vars
    for k in ["PALS_RANKID", "PALS_SIZE", "PALS_LOCAL_RANKID", "PALS_LOCAL_SIZE", "PMIX_RANK"]:
         print(f"{k}={os.environ.get(k)}")

subprocess.run([sys.executable] + sys.argv[1:])
