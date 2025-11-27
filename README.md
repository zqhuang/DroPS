Deriving r from Power Spectra (DroPS)

This is a quick-start documentation. For more details see doc/DroPS_doc_en.pdf


The software can
1. Generate reference simulations of CMB/noise/foreground;
2. Simulate sky maps;
3. Calculate TT/TE/TB/EE/EB/BB band powers of masked maps;
4. Measure cosmological and foreground parameters from masked sky maps and reference simulations.
5. Do component separation when $\beta_s$ and $\beta_d$ are known.


INSTALL:

Set up a virtual environment and activate it
(please google "python virtual environment" to learn how to do that)

upgrade pip for the latest info of packages
pip install --upgrade pip

install dependences
pip install -r requirements.txt

Modify pysm3 by replacing path_to_pysm3/models/cmb.py (where pyth_to_pysm3 is the local path where pysm3 is installed) with cmb.py in this repository.


Main functions:

1. generate a TOD filtering model by running

python mock_filtering.py
enter the nside (128 for testing, 256/512 for serious simulations) and file name (e.g. filter_128.pickle)

2. simulate cmb/noise/foreground maps with a 4-channel ground-based experiment

python simulate.py Test/test_sim_config.txt


3. simulate  sky maps

python simulate.py Test/test_sim_config.txt maps/test_  0.01 9999

you can replace maps/test_ with your preferred prefix of the output maps, 0.01 with your preferred fiducial r, and 9999 with your preferred random seed

4. analyse the sky maps and obtain constraint on r

python mainpipe.py Test/test_ana_config.txt maps/test_

