#!/bin/bash

output="output_test_stp/"
domain_name="5x5-stp-"

heuristic_scheme=("--learned-heuristic") 
algorithm="GBFS"
problems_dir="problems/stp/puzzles_5x5_test/"

for iter in {1..5}; do
	for scheme in "${heuristic_scheme[@]}"; do
		lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
		name_scheme=${scheme// /}
		name_scheme=${name_scheme//-heuristic/}
		name_scheme=${name_scheme//--/-}
		output_exp="${output}${lower_algorithm}${name_scheme}-v${iter}"
		model=${domain_name}${lower_algorithm}${name_scheme}-v${iter}
			
		sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir} run_bootstrap_test_stp.sh
	done
done
