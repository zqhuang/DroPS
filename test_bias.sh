for i in `seq 20` 
do 
    python simulate.py Test/test_sim_config.txt maps/test${i}_  0.01 ${i}  d1s1  
    python mainpipe.py Test/test_ana_config.txt maps/test${i}_ Test/r1_logfile_d1s1.txt
done
