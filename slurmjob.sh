#!/usr/bin/zsh

#SBATCH -J gpu_serial
#SBATCH -o gpu_serial.%J.log
#SBATCH --gres=gpu:1

### Start of Slurm SBATCH definitions
# Ask for eight tasks (same as 8 cores for this example)
###SBATCH --ntasks=2
#--ntasks-per-node=2

# Ask for the maximum memory per CPU (less than 4GB)
#SBATCH --mem-per-cpu=16GB

# Ask for up to 15 Minutes of runtime
#SBATCH --time=6:00:00

# Name the job
#SBATCH --job-name=myjob

# Declare a file where the STDOUT/STDERR outputs will be written
#SBATCH --output=logs/output.%J.txt

### end of Slurm SBATCH definitions

##SBATCH --account=thes1729
#SBATCH -p c23g
 


### your program goes here (hostname is an example, can be any program)
# `srun` runs `ntasks` instances of your programm `hostname`
module load GCCcore/.9.3.0
# module load Python/3.9.6
module load cuDNN/8.6.0.163-CUDA-11.8.0

export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate thesisenv
python run_imputation.py --config config/grin/traffic_block.yaml --dataset-name traffic_block --in-sample False
