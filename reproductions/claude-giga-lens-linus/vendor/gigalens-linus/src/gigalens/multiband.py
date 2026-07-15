import numpy as np
from astropy.io import fits
from gigalens.prior import Prior

class SourcePlane:
    
    def __init__(self, 
                 prior: Prior, 
                 path=None, 
                 observed_image=None, 
                 error_map=None, 
                 mask=None, 
                 background_rms=None, 
                 exp_time=None, 
                 psf=None
                ):
        
        if path is not None:
            with fits.open(path) as hdul:
                if observed_image is None: self.observed_image = hdul['DATA'].data
                if error_map is None: self.error_map = np.sqrt(hdul['STAT'].data)
                if background_rms is None: self.background_rms = hdul['DATA'].header['BKG_RMS']
                if exp_time is None: self.exp_time = hdul['PRIMARY'].header['EXPTIME']
        else:
            self.observed_image=observed_image
            self.error_map=error_map
            self.background_rms=background_rms
            self.exp_time=exp_time

        self.psf=psf
        self.mask=mask
        self.prior=prior


def get_from_source_list(source_list: list[SourcePlane], psf=None, exp_time=None):
    data = dict(
        source_priors = [source.prior for source in source_list],
        observed_images = [source.observed_image for source in source_list],
        error_maps = [source.error_map for source in source_list],
        masks = [source.mask for source in source_list],
        background_rmss = [source.background_rms for source in source_list],
        exp_times = [source.exp_time for source in source_list] if exp_time is None else exp_time,
        psfs = [source.psf for source in source_list] if psf is None else psf,
    )
    return data