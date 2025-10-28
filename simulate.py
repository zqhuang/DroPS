from beforecmb import sky_simulator
from sys import argv

def str2models(s):
    i = 1
    while(s[i] != 'd' and s[i] != 's'):
        i += 1
    return [s[0:i], s[i:]]

if(len(argv) < 3):
    sim = sky_simulator(config_file=argv[1])
    sim.simulate_noise()
    sim.simulate_cmb()
    sim.simulate_cmb1()
    sim.simulate_fg()
else:
    if(argv[2] == 'DEFAULT'):
        sim = sky_simulator(config_file=argv[1])
    else:
        sim = sky_simulator(config_file=argv[1], root_overwrite=argv[2])
    print("root = ", sim.root)
    if(len(argv)<4):
        sim.simulate_map()        
    elif(len(argv) < 5):
        print('using fiducial r = ', argv[3])
        sim.simulate_map(r=float(argv[3]))
    elif(len(argv)<6):
        print('using fiducial r = ', argv[3])        
        print('using seed: ' + argv[4])
        sim.simulate_map(r=float(argv[3]), seed = int(argv[4]))
    else:
        fgmodels = str2models(argv[5])
        sim.fg_models = fgmodels        
        print('using fiducial r = ', argv[3])        
        print('using seed: ' + argv[4])
        print('foreground models: ', sim.fg_models)        
        sim.simulate_map(r=float(argv[3]), seed = int(argv[4]))        
