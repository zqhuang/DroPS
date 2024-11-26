from beforecmb import sky_simulator
from sys import argv

if(len(argv) < 3):
    sim = sky_simulator(config_file=argv[1])
    sim.simulate_noise()
    sim.simulate_cmb()
    sim.simulate_cmb1()
    sim.simulate_fg()
else:
    sim = sky_simulator(config_file=argv[1], root_overwrite=argv[2])
    print("root = ", sim.root)
    if(len(argv)<4):
        sim.simulate_map()        
    else:
        sim.simulate_map(r=float(argv[3]))
    

