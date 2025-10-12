import numpy as np
import pysm3
import healpy as hp
from astropy.utils.data import import_file_to_cache, export_download_cache, is_url_in_cache, is_url


def export_file(cache_filename, local_filename):
    url = r'https://portal.nersc.gov/project/cmb/pysm-data/pysm_2/'+cache_filename    
    if(not is_url):
        print("Failed exporting:" + cache_filename + " => " + local_filename)
        print("Not a valid URL")
        exit()
    if(not is_url_in_cache(url)):
        print("Failed exporting:" + cache_filename + " => " + local_filename)
        print("Cannot find the file in cache.")
        exit()        
    export_download_cache(local_filename, url)


def import_file(local_filename, cache_filename):
    url = r'https://portal.nersc.gov/project/cmb/pysm-data/pysm_2/'+cache_filename
    if(not is_url):
        print("Failed importing:" + local_filename + " => " + cache_filename)
        print("Not a valid URL")
        exit()
    if(is_url_in_cache(url)):
        print("Failed importing:" + local_filename + " : " + cache_filename)
        print("A file is already in cache, should not overwrite.")
        exit()        
    import_file_to_cache(url, local_filename) 
    
local_dir = r"/home/zqhuang/work/pysm_2/"
    
beta_file = hp.smoothing(hp.read_map(local_dir + "dust_beta.fits"), fwhm=np.pi/180.*2.)
temp_file = hp.smoothing(hp.read_map(local_dir + "dust_temp.fits"), fwhm=np.pi/180.*2.)
hp.write_map(local_dir + "dust_beta_2deg.fits", beta_file)
hp.write_map(local_dir + "dust_temp_2deg.fits", temp_file)
import_file(local_dir + "dust_beta_2deg.fits", "dust_beta_2deg.fits")
import_file(local_dir + "dust_temp_2deg.fits", "dust_temp_2deg.fits")
