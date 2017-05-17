#!/bin/sh


for out in -1;do	#-1 -2 -5 -10 -15 -20 
	echo "$out"


	for bs in 2 5 10;do
		
		for layers in 1 ;do	
			for cells_hidden in 5 6 7 8 9 10;do
				for n_iters in 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000;do	
					python rnn_lstm.py $bs $layers $cells_hidden $n_iters $out
		
done
done
done
done
done
