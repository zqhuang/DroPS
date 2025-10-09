#!/bin/bash
for i in `seq 50`
do
    python simulate.py Test/test_sim_d0s0.txt maps/r0test${i}_ 0. $i    
    python mainpipe.py Test/test_ana_d0s0.txt maps/r0test${i}_
done
	 
