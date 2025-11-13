DroPS
Deriving r from Power Spectra (DroPS):  measuring primordial gravitational wave (tensor-to-scalar ratio r) from CMB maps.

This is a quick-start documentation. For more details see doc/DroPS_doc_en.pdf

INSTALL:
--------------------------------------------------------
Set up a virtual environment and activate it
(please google "python virtual environment" to learn how to do that)

upgrade pip for the latest info of packages
pip install --upgrade pip

install dependences
pip install -r requirements.txt

hack pysm3 by replacing path_to_pysm3/models/cmb.py (where pyth_to_pysm3 is the local path where pysm3 is installed) with cmb.py in this repository. (use "whereis python" to find the python path)
 This step is for more efficient simulations of CMB maps.
 If you only want to analyze maps, you can skip this step
----------------------------------------------------------------




Simulating and analysing sky maps:

----------------------------------------------------------------

1. generate a TOD filtering model by running

python mock_filtering.py
enter the nside (128 for testing, 256/512 for serious simulations) and file name (e.g. filter_128.pickle)

2. simulate cmb/noise/foreground maps with a 4-channel ground-based experiment

python simulate.py Test/test_sim_config.txt

------------------ single simulation -----------------------------------

3a. simulate one set of sky maps

python simulate.py Test/test_sim_config.txt maps/test_  0.01 9999

you can replace maps/test_ with your preferred prefix of the output maps, 0.01 with your preferred fiducial r, and 9999 with your preferred random seed

4a. analyse the sky maps and obtain constraint on r

python mainpipe.py Test/test_ana_config.txt maps/test_


------------testing bias by running multiple simulations--------------

3b. simulate multiple sets of sky maps

You need to simulate more than one set of sky maps if you want to test the bias of r measurement. You may use the shell script to simulate 20 sets of sky maps with different random seeds. 
./sim.sh

4b. measure r from multiple sets of sky maps

If you want to measure r for the maps generated with sim.sh, you can run:
./ana.sh

5b. plot the results
python plot_rs.py Test/r_logfile.txt 0.01
