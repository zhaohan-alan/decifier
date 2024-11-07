#!/bin/bash
# #SBATCH -p gpu --gres=gpu:a100:1
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 0-01:00:00
#SBATCH --job-name=eval_decifer
# #SBATCH --array 0-11%12
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=6G
#SBATCH --output=logs/experiment_%A_%a.out

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
  "--dataset-name noprompt_0pctnoise_1pctfwhm"
)
#  "--dataset-name composition_0pctnoise_1pctfwhm --add-composition"
#  "--dataset-name composition_spacegroup_0pctnoise_1pctfwhm --add-composition --add-spacegroup"
#  "--dataset-name composition_spacegroup_1pptnoise_1pctfwhm --add-composition --add-spacegroup --add-noise 0.001 --add-broadening 0.01"
#  "--dataset-name composition_spacegroup_1pctnoise_1pctfwhm --add-composition --add-spacegroup --add-noise 0.01 --add-broadening 0.01"
#  "--dataset-name composition_spacegroup_5pctnoise_1pctfwhm --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.01"
#  "--dataset-name composition_spacegroup_1pptnoise_10pctfwhm --add-composition --add-spacegroup --add-noise 0.001 --add-broadening 0.1"
#  "--dataset-name composition_spacegroup_1pctnoise_10pctfwhm --add-composition --add-spacegroup --add-noise 0.01 --add-broadening 0.1"
#  "--dataset-name composition_spacegroup_5pctnoise_10pctfwhm --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.1"
#  "--dataset-name composition_spacegroup_1pptnoise_50pctfwhm --add-composition --add-spacegroup --add-noise 0.001 --add-broadening 0.5"
#  "--dataset-name composition_spacegroup_1pctnoise_50pctfwhm --add-composition --add-spacegroup --add-noise 0.01 --add-broadening 0.5"
#  "--dataset-name composition_spacegroup_5pctnoise_50pctfwhm --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.5"
#)

ARRAY_ARGS=${ARGS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Display the arguments
echo "Model and data arguments passed: ${EXTERNAL_ARGS[*]}"
echo "Experimental arguments passed: $ARGS_ARRAY"

python bin/evaluate.py $ARRAY_ARGS "${EXTERNAL_ARGS[@]}"
