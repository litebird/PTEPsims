import healpy as hp
import numpy as np
import pysm3
import pysm3.units as u
import argparse
import importlib.util
import os
import ptep_mbs.instrument
from ptep_mbs.utils import *
from tqdm import tqdm

class ForegroundModel:

    def __init__(self,params):
        self.params = params
        self.instr = getattr(ptep_mbs.instrument, params.inst)
        self.nside = params.nside
        self.npix = hp.nside2npix(self.nside)
        self.channels = self.instr.keys()
        self.fg_models = params.fg_models
        self.components = list(self.fg_models.keys())
        self.ncomp = len(self.components)
        self.smooth = params.gaussian_smooth
        self.libdir = params.fg_dir
        os.makedirs(self.libdir, exist_ok=True)
    
    def fg_sims_comp(self,cmp):
        fg_config_file_name = self.fg_models[cmp]
        if ('lb' in fg_config_file_name) or ('pysm' in fg_config_file_name):
            fg_config_file_path = os.path.join(
                os.path.dirname(__file__), 'fg_models')
            fg_config_file = os.path.join(fg_config_file_path,fg_config_file_name)
        else:
            fg_config_file = f'{fg_config_file_name}'
        sky = pysm3.Sky(nside=self.nside, component_config=fg_config_file)
        fg_maps = []
        for chnl in tqdm(self.channels, desc=f'Foreground simulation for component {cmp}',leave=True,unit='channel'):
            fname = os.path.join(self.libdir,f'{chnl}_{cmp}_{self.nside}.fits')
            if os.path.exists(fname):
                fg_maps.append(hp.read_map(fname, (0,1,2), verbose=False))
                continue

            freq = self.instr[chnl]['freq']
            fwhm = self.instr[chnl]['beam']
            if self.params.band_int:
                band = self.instr[chnl]['freq_band']
                fmin = freq-band/2.
                fmax = freq+band/2.
                fsteps = fmax-fmin+1
                bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                weights = np.ones(len(bandpass_frequencies))
                sky_extrap = sky.get_emission(bandpass_frequencies, weights)
                sky_extrap = sky_extrap*bandpass_unit_conversion(bandpass_frequencies, weights, u.uK_CMB)
            else:
                sky_extrap = sky.get_emission(freq*u.GHz)
                sky_extrap = sky_extrap.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
            if self.smooth:
                sky_extrap_smt = hp.smoothing(sky_extrap, fwhm = np.radians(fwhm/60.), verbose=False)
            else:
                sky_extrap_smt = sky_extrap
            fg_maps.append(sky_extrap_smt)
            hp.write_map(fname, sky_extrap_smt, dtype=np.float32)
        return np.array(fg_maps)
    
    def fg_sims(self):
        for i, cmp in enumerate(self.components):
            if i==0:
                fg_maps = self.fg_sims_comp(cmp)
            else:
                fg_maps += self.fg_sims_comp(cmp)
        return fg_maps

        



def make_fg_sims(params):
    """ Write foreground maps on disk

    Parameters
    ----------
    params: module contating all the simulation parameters

    """
    parallel = params.parallel
    instr = getattr(ptep_mbs.instrument, params.inst)
    nside = params.nside
    smooth = params.gaussian_smooth
    root_dir = params.out_dir
    out_dir = f'{root_dir}/foregrounds/'
    file_str = params.file_string
    channels = instr.keys()
    fg_models = params.fg_models
    components = list(fg_models.keys())
    ncomp = len(components)
    rank = 0
    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        rank_to_use = list(range(ncomp))
    if not os.path.exists(out_dir) and rank==0:
        os.makedirs(out_dir)
    if rank==0:
        for cmp in components:
            if not os.path.exists(out_dir+cmp) and rank==0:
                os.makedirs(out_dir+cmp)
            fg_config_file_name = fg_models[cmp]
            if ('lb' in fg_config_file_name) or ('pysm' in fg_config_file_name):
                fg_config_file_path = os.path.join(
                    os.path.dirname(__file__), 'fg_models/')
                fg_config_file = f'{fg_config_file_path}/{fg_config_file_name}'
            else:
                fg_config_file = f'{fg_config_file_name}'
            sky = pysm3.Sky(nside=nside, component_config=fg_config_file)
            for chnl in channels:
                freq = instr[chnl]['freq']
                fwhm = instr[chnl]['beam']
                if params.band_int:
                    band = instr[chnl]['freq_band']
                    fmin = freq-band/2.
                    fmax = freq+band/2.
                    fsteps = fmax-fmin+1
                    bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                    weights = np.ones(len(bandpass_frequencies))
                    sky_extrap = sky.get_emission(bandpass_frequencies, weights)
                    sky_extrap = sky_extrap*bandpass_unit_conversion(bandpass_frequencies, weights, u.uK_CMB)
                else:
                    sky_extrap = sky.get_emission(freq*u.GHz)
                    sky_extrap = sky_extrap.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
                if smooth:
                    sky_extrap_smt = hp.smoothing(sky_extrap, fwhm = np.radians(fwhm/60.), verbose=False)
                else:
                    sky_extrap_smt = sky_extrap
                if rank==0:
                    file_name = f'{chnl}_{cmp}_{file_str}.fits'
                    file_tot_path = f'{out_dir}{cmp}/{file_name}'
                    hp.write_map(file_tot_path, sky_extrap_smt, overwrite=True, dtype=np.float32)
