#!/bin/bash
for i in `seq 20`
do
    python mainpipe.py Test/test_ana_config.txt maps/test${i}_r100E-4_ 
done
	 
