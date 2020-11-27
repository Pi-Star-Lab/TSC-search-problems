#!/bin/bash

declare -a losses=("CrossEntropyLoss" "LevinLoss" "ImprovedLevinLoss")
#declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss" "RegLevinLoss")
output="output_train_sokoban/"
domain_name="10x10-sokoban-"
algorithm="LevinStar"

scheduler="pgbs"
heuristic_scheme=("--learned-heuristic")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic")

for iter in {2..2}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-v${iter}"
			model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-v${iter}

			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},loss=${loss},model=${model},scheduler=${scheduler} run_bootstrap_train_sokoban.sh
		done
	done
done
