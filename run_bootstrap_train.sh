#!/bin/bash
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=8000M        # memory per node
#SBATCH --time=0-11:59      # time (DD-HH:MM)
#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID
#SBATCH --account=def-lelis

module load python/3.6
source tensorflow/bin/activate
python src/main.py ${scheme} -a ${algorithm} -l ${loss} -m ${model} -p problems/stp/puzzles_4x4_train/ --learn -d SlidingTile -b 10000

