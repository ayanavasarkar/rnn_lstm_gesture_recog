#!/bin/bash


START=0
END=39

for (( c=$START; c<=$END; c++ ));
do
	echo  "$c "

	for bs in 2 ;do
		
		for layers in 1 ;do	
			
			for cells_hidden in 10;do
				
				for n_iters in 10100;do	
					python rnn_lstm_test.py $bs $layers $cells_hidden $n_iters $c
		
done
done
done
done
done
