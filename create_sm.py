import numpy as np
import pysm3
import healpy as hp
from astropy.utils.data import import_file_to_cache, export_download_cache, is_url_in_cache, is_url, get_cached_urls, cache_contents


url_prefix = r'https://portal.nersc.gov/project/cmb/pysm-data/'

def export_file(cache_filename, local_filename):
    theurl = url_prefix + cache_filename    
    if(not is_url(theurl)):
        print("Failed exporting:" + cache_filename + " => " + local_filename)
        print("Not a valid URL")
        exit()
    if(not is_url_in_cache(theurl)):
        print("Failed exporting:" + cache_filename + " => " + local_filename)
        print("Cannot find the file in cache.")
        exit()        
    export_download_cache(local_filename, theurl)


def import_file(local_filename, cache_filename):
    url =  url_prefix + cache_filename
    if(not is_url(url)):
        print("Failed importing:" + local_filename + " => " + cache_filename)
        print("Not a valid URL")
        exit()
    if(is_url_in_cache(url)):
        print("Failed importing:" + local_filename + " : " + cache_filename)
        print("A file is already in cache, should not overwrite.")
        exit()        
    import_file_to_cache(url, local_filename) 
    
local_dir = r"/home/zqhuang/work/pysm_2/"

#filemap = cache_contents()
#theurl = url_prefix + r"synch/synch_beta_nside2048_2023.02.16.fits"
#print(filemap[theurl])

beta_file = hp.smoothing(hp.read_map(r"s5_beta_map.fits"), fwhm=np.pi/180.*2.)
hp.write_map("sm_beta_map.fits", beta_file)
import_file("sm_beta_map.fits", "pysm_2/sm_synch_beta_2deg.fits")

