#!/bin/bash
for i in `seq 20`
do
    python mainpipe.py Test/test_ana_config.txt maps/test${i}_
done
	 
