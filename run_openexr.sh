#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH -p gpu-he --gres=gpu:1
#SBATCH -n 8
#SBATCH -N 1 
#SBATCH --mem=5GB
#SBATCH -J openexr
##SBATCH -C quadrortx
##SBATCH --constraint=v100
#SBATCH -o /users/aarjun1/data/aarjun1/prj_depth/logs/MI_%A_%a_%J.out
#SBATCH -e /users/aarjun1/data/aarjun1/prj_depth/logs/MI_%A_%a_%J.err
##SBATCH --account=carney-tserre-condo
##SBATCH --array=0-25

##SBATCH -p gpu

cd /users/aarjun1/data/aarjun1/prj_depth/

# ############################################
module load python/3.5.2
module load openexr/2.2.1

module load opencv-python
module load matplotlib/2.2.4

# module load imageio
# module load anaconda/3-5.2.0

source ~/ENV/bin/activate

############################################
# module load anaconda/3-5.2.0
# module load python/3.5.2
# source activate /users/aarjun1/anaconda/stim_gen

############################################

echo $SLURM_ARRAY_TASK_ID

python openexr_reader.py
# python file_management.py

