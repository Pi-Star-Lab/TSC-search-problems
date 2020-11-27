#!/bin/bash

declare -a losses=("CrossEntropyLoss")
#declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss" "RegLevinLoss")
output="output_test_stp/"
domain_name="4x4-stp-"
problems_dir="problems/stp/puzzles_4x4_test/"

heuristic_scheme=("")
#heuristic_scheme=("--learned-heuristic" "")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic" "")
algorithm="Levin"

for iter in {3..3}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}"
			model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-v${iter}

			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir} run_bootstrap_test_stp.sh
		done
	done
done
