# DroPS
Deriving r from Power Spectra (DroPS) of CMB


please install all dependences:
camb
healpy
astropy
pymaster 
pysm3
camb
matplotlib
numpy
emcee (this is actually not used by default)

To simulate sky maps, you need to

1. hack pysm3 by replacing path_to_pysm3/models/cmb.py (where pyth_to_pysm3 is the local path where pysm3 is installed) with cmb.py in this repository.

2. generate a TOD filtering model by running

python mock_filtering.py
enter the nside (128 for testing, 256/512 for serious simulations) and file name (e.g. filter_128.pickle)

3. simulate maps with a 4-channel ground-based experiment

python simulate.py Test/test_sim_config.txt


python simulate.py Test/test_sim_config.txt testmap 0.01



#=======================================================
Example of simulating CMB-S4 sky maps:

python simulate.py CMBS4/cmbs4_sim_config.py

if you want to simulate the data map with a different fiducial r, say r=0.01, you can run

python simulate.py CMBS4/cmbs4_sim_config.py  new_root_name  0.01


Example of analysing the data maps and inferring r

python mainpipe.py CMBS4/cmbs4_ana_config.py 
#===================================================