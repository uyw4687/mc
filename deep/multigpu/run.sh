#!/bin/bash

#SBATCH --job-name=colorizer     # Submit a job named "colorizer"
#SBATCH --nodes=1                # Using 1 node
#SBATCH --gpus-per-node=4        # Using 4 GPUs per node
#SBATCH --time=0-00:15:00        # 15 minute timelimit
#SBATCH --mem=16000MB            # Using 16GB memory per node
#SBATCH --exclusive              # Take node exclusively

srun ./main "$@"
