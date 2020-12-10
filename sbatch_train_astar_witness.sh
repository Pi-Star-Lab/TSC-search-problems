#!/bin/bash

output="output_train_witness/"
domain_name="4x4-witness50k-"
algorithm="AStar"
loss="MSELoss"
heuristic_scheme=("--learned-heuristic")

scheduler="online"

for iter in {1..5}; do
	for weight in 1.0 1.5 2.0 2.5 3.0; do
		for scheme in "${heuristic_scheme[@]}"; do
			lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
			name_scheme=${scheme// /}
			name_scheme=${name_scheme//-heuristic/}
			name_scheme=${name_scheme//--/}
			
			model=${domain_name}${lower_algorithm}-${name_scheme}-${scheduler}-w${weight//./}-v${iter}
			output_exp="${output}${lower_algorithm}-${name_scheme}-${scheduler}-w${weight//./}-v${iter}"
	
			#echo ${model}
			#echo ${output_exp}
	
			sbatch --output=${output_exp} --export=scheme="${scheme}",algorithm=${algorithm},loss=${loss},model=${model},scheduler=${scheduler},weight=${weight} run_bootstrap_train_astar_witness.sh
		done
	done
done
