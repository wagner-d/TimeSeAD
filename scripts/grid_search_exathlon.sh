#!/bin/bash

# Script to run grid search for Exathlon dataset
# Links to the log files are stored in a results directory
# Sample call
# ./grid_search_exathlon.sh <result_name> <path_to_grid_config> [<specific app_id>]


RESULTS_NAME=$1
TEST_CONFIG=$2
SEEDS=(1 123 111)
APP_IDS=(1 2 3 4 5 6 9 10)

if [ ! -f $TEST_CONFIG ]; then
    echo "Config file $TEST_CONFIG not found!"
    exit
fi

if [ ! -z "$3" ]; then
    APP_IDS=($3)
fi

for app_id in "${APP_IDS[@]}"
do
    for seed in "${SEEDS[@]}"
    do
        echo "================================================="
        echo "Running experiment with seed=$seed app_id=$app_id"
        echo "================================================="

        python timesead_experiments/grid_search.py with $TEST_CONFIG \
            "dataset.ds_args.app_id=$app_id" \
            "training_param_updates.dataset.ds_args.app_id=$app_id" \
            "training_param_updates.training.deterministic=True" \
            "training_param_updates.training.device=cuda" \
            "seed=$seed"

        # Get numerically sorted folder list, and get the second folder. (first folder is _sources)
        log_dir=$(ls -dv1r log/grid_search/* | awk 'NR==2')
        
        if [ ! -f $log_dir/info.json ]; then
            echo "Run might've failed, info.json missing for $log_dir"
            exit
        fi

        mkdir -p results/exathlon_${app_id}

        link_file=results/exathlon_${app_id}/${RESULTS_NAME}_${seed}
        rm $link_file
        ln -srf ${log_dir} $link_file
    done
done

