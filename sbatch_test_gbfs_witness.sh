#!/bin/bash

output="output_test_witness/"
domain_name="4x4-witness-"

heuristic_scheme=("--learned-heuristic")
algorithm="GBFS"
problems_dir="problems/witness/puzzles_4x4_test"

scheduler="online"

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
		name_scheme=${scheme// /}
		name_scheme=${name_scheme//-heuristic/}
		name_scheme=${name_scheme//--/-}
		output_exp="${output}${lower_algorithm}${name_scheme}-${scheduler}-v${iter}"
		model=${domain_name}${lower_algorithm}-${name_scheme}-${scheduler}-v${iter}
		
		#echo ${output_exp}
		#echo ${model}	
		sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},model=${model},problem=${problems_dir} run_bootstrap_test_witness.sh
	done
done
