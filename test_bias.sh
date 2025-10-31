for i in `seq 100` 
do 
    python simulate.py AliCPT/AliCPT_sim_config.txt maps/AliCPT_r1_${i}_  0.01 ${i}  d1s1  
    python mainpipe.py AliCPT/AliCPT_ana_config.txt maps/AliCPT_r1_${i}_ AliCPT/r1_logfile_d1s1.txt
done
