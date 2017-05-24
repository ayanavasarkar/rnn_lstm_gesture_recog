#!/bin/bash



START=-40
END=-1

for (( c=$START; c<=$END; c++ ));
do
	echo  "$c "

	python rnn_lstm_testing.py 5 4 256 1000 $c
		

done
