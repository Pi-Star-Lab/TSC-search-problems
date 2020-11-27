#!/bin/bash

declare -a losses=("CrossEntropyLoss" "ImprovedLevinLoss" "LevinLoss")
output="output_test_witness/"
domain_name="4x4-witness-"
problems_dir="problems/witness/puzzles_4x4_test"

heuristic_scheme=("--learned-heuristic")
algorithm="LevinStar"

scheduler="online"
#mix_epsilon="0.01"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for loss in ${losses[@]}; do
			lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/-}
			#output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-mix${mix_epsilon//./}-v${iter}"
			output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-v${iter}"
			model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-v${iter}

			#echo ${output_exp}
			#echo ${model}

			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir},mix_epsilon=${mix_epsilon} run_bootstrap_test_witness.sh
		done
	done
done
