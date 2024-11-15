#!/bin/bash
#SBATCH -p gpu --gres=gpu:titanrtx:1
#SBATCH --time 1-00:00:00
#SBATCH --job-name=eval_deCIFer
# #SBATCH --array 0-8%9
#SBATCH --array 0
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=12G
#SBATCH --output=logs/exp_cond_%A_%a.out

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
#  "--dataset-name noprompt_0pctnoise_5pctfwhm"
  "--dataset-name composition_0pctnoise_5pctfwhm --add-composition"
#  "--dataset-name composition_spacegroup_0pctnoise_5pctfwhm --add-composition --add-spacegroup"
#  "--dataset-name composition_spacegroup_1pptnoise_5pctfwhm --add-composition --add-spacegroup --add-noise 0.001 --add-broadening 0.05"
#  "--dataset-name composition_spacegroup_1pctnoise_5pctfwhm --add-composition --add-spacegroup --add-noise 0.01 --add-broadening 0.05"
#  "--dataset-name composition_spacegroup_5pctnoise_5pctfwhm --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.05"
#  "--dataset-name composition_spacegroup_1pptnoise_10pctfwhm --add-composition --add-spacegroup --add-noise 0.001 --add-broadening 0.1"
#  "--dataset-name composition_spacegroup_1pctnoise_10pctfwhm --add-composition --add-spacegroup --add-noise 0.01 --add-broadening 0.1"
#  "--dataset-name composition_spacegroup_5pctnoise_10pctfwhm --add-composition --add-spacegroup --add-noise 0.05 --add-broadening 0.1"
)

ARRAY_ARGS=${ARGS_ARRAY[$SLURM_ARRAY_TASK_ID]}

# Display the arguments
echo "Model and data arguments passed: ${EXTERNAL_ARGS[*]}"
echo "Experimental arguments passed: $ARRAY_ARGS"

python bin/evaluate.py $ARRAY_ARGS "${EXTERNAL_ARGS[@]}" --clean_fwhm 0.05 --default_fwhm 0.05 --debug-max 5000
