#!/bin/bash
#SBATCH -p gpu --gres=gpu:a100:1
<<<<<<< HEAD:decifer/slurm/train_cl.sh
#SBATCH --time 0-12:00:00
#SBATCH --job-name=cl_decifer
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/cl_%A_%a.out
=======
#SBATCH --time 3-00:00:00
#SBATCH --job-name=train_CL
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/train_cl_%A_%a.out
>>>>>>> augment_at_train:slurm/train_cl.sh

# Function to display help message
usage() {
  echo "Usage: $0 [options]"
  echo "Pass any number of arguments and their values to the script, e.g. --a value1 --b value2"
  exit 1
}

# Check if any arguments are provided
if [ "$#" -eq 0 ]; then
  usage
fi

# Collect all aguments
ARGS=("$@")

# Display the arguments
echo "Arguments passed: ${ARGS[*]}"

<<<<<<< HEAD:decifer/slurm/train_cl.sh
python scripts/train_cl_embeddings.py "${ARGS[@]}"
=======
python bin/train_cl_embeddings.py "${ARGS[@]}"
>>>>>>> augment_at_train:slurm/train_cl.sh
