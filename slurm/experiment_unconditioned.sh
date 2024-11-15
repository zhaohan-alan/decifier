#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=exp_deCIFer
# #SBATCH --array 0-2%3
#SBATCH --array 0-1%2
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/exp_uncond_%A_%a.out

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
EXTERNAL_ARGS=("$@")

# Define specific argument sets for each array job
ARGS_ARRAY=(
#  "--dataset-name noprompt"
  "--dataset-name composition --add-composition"
  "--dataset-name composition_spacegroup --add-composition --add-spacegroup"
)

ARRAY_ARGS=${ARGS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Display the arguments
echo "Model and data arguments passed: ${EXTERNAL_ARGS[*]}"
echo "Experimental arguments passed: $ARRAY_ARGS"

python bin/evaluate.py $ARRAY_ARGS "${EXTERNAL_ARGS[@]}" --clean_fwhm 0.05 --default_fwhm 0.05 --debug-max 5000
