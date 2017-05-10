#!/bin/sh


for out in -1 -2 -5 -10 -15 -20 -30;do
	echo "$out"


	for bs in 2 5 10;do
		
		for layers in 1 2 3 4;do	
			for cells_hidden in 128 256 512;do
				for n_iters in 100 500 1000;do	
					python rnn_lstm_optimized.py $bs $layers $cells_hidden $n_iters $out
		
done
done
done
done
done
