#!/bin/bash

output="output_train_sokoban/"
domain_name="10x10-sokoban-"
algorithm="PUCT"
declare -a losses=("CrossEntropyLoss")
constants=("1.5") 
heuristic_scheme=("--learned-heuristic")
#heuristic_scheme=("--learned-heuristic --default-heuristic" "--default-heuristic" "--learned-heuristic")

scheduler="online" 

for iter in {1..1}; do
	for scheme in "${heuristic_scheme[@]}"; do
		for c in "${constants[@]}"; do
			for loss in ${losses[@]}; do
				lower_loss=$(echo ${loss} | tr "A-Z" "a-z")
				lower_algorithm=$(echo ${algorithm} | tr "A-Z" "a-z")
				name_scheme=${scheme// /}
				name_scheme=${name_scheme//-heuristic/}
				name_scheme=${name_scheme//--/-}
				c_name=${c//./}
				name_scheme=${name_scheme//--/-}
				output_exp="${output}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-c${c_name}-v${iter}"
				model=${domain_name}${lower_algorithm}-${lower_loss}${name_scheme}-${scheduler}-c${c_name}-v${iter}
		
				sbatch --output=${output_exp} --export=scheme="${scheme}",constant=${c},algorithm=${algorithm},loss=${loss},model=${model},scheduler=${scheduler} run_bootstrap_train_sokoban.sh
			done
		done
	done
done
