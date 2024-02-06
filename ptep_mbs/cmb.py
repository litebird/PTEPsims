import healpy as hp
import numpy as np
import argparse
import importlib.util
import os
import math
import ptep_mbs.instrument
import pysm3
import pysm3.units as u
from ptep_mbs.utils import *
from pysm3.models.template import Model
from pysm3 import utils
from tqdm import tqdm


class CMBMap(Model):
    """Load one or a set of 3 CMB maps"""

    def __init__(self, nside, map_IQU=None):
        super().__init__(nside=nside)
        if map_IQU is not None:
            assert len(map_IQU) == 3, "map_IQU must be a list of 3 maps"
            self.map = map_IQU
            self.map = u.Quantity(self.map, unit=u.uK_CMB)
        else:
            raise (ValueError("No input map provided"))

    @u.quantity_input
    def get_emission(self, freqs: u.GHz, weights=None) -> u.uK_RJ:
        freqs = utils.check_freq_input(freqs)
        # do not use normalize weights because that includes a transformation
        # to spectral radiance and then back to RJ
        if weights is None:
            weights = np.ones(len(freqs), dtype=np.float32)

        scaling_factor = utils.bandpass_unit_conversion(
            freqs * u.GHz, weights, output_unit=u.uK_RJ, input_unit=u.uK_CMB
        )

        return u.Quantity(self.map * scaling_factor, unit=u.uK_RJ, copy=False)

class CMBModel:

    def __init__(self,params):
        self.params = params
        self.instr = getattr(ptep_mbs.instrument, params.inst)
        self.nside = params.nside
        self.npix = hp.nside2npix(self.nside)
        self.channels = self.instr.keys()
        self.smooth = params.gaussian_smooth
        self.seed_cmb = params.seed_cmb
        self.cmb_ps_file = params.cmb_ps_file

        if self.cmb_ps_file:
            print(self.cmb_ps_file)
            cl_cmb = hp.read_cl(self.cmb_ps_file)
        else:
            cmb_ps_scalar_file = os.path.join(
                os.path.dirname(__file__),
                'datautils/Cls_Planck2018_for_PTEP_2020_r0.fits')
            self.cl_cmb_scalar = hp.read_cl(cmb_ps_scalar_file)
            cmb_ps_tensor_r1_file = os.path.join(
                os.path.dirname(__file__),
                'datautils/Cls_Planck2018_for_PTEP_2020_tensor_r1.fits')
            self.cmb_r = params.cmb_r
            self.cl_cmb_tensor = hp.read_cl(cmb_ps_tensor_r1_file)*self.cmb_r
            self.cl_cmb = self.cl_cmb_scalar+self.cl_cmb_tensor
        
    def cmb_sims(self,idx):
        if self.seed_cmb:
            np.random.seed(self.seed_cmb+idx)
        cmb_temp = hp.synfast(self.cl_cmb, self.nside, new=True, verbose=False)
        
        sky = pysm3.Sky(nside=self.nside, component_objects=[CMBMap(self.nside, map_IQU=cmb_temp)])
        cmb_maps = []
        for chnl in tqdm(self.channels, desc=f'CMB simulation',leave=True,unit='channel'):
            freq = self.instr[chnl]['freq']
            if self.params.band_int:
                band = self.instr[chnl]['freq_band']
                fmin = freq-band/2.
                fmax = freq+band/2.
                fsteps = fmax-fmin+1
                bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                weights = np.ones(len(bandpass_frequencies))
                cmb_map = sky.get_emission(bandpass_frequencies, weights)
                cmb_map = cmb_map*bandpass_unit_conversion(bandpass_frequencies, weights, u.uK_CMB)
            else:
                cmb_map = sky.get_emission(freq*u.GHz)
                cmb_map = cmb_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
            fwhm = self.instr[chnl]['beam']
            if self.smooth:
                cmb_map_smt = hp.smoothing(cmb_map, fwhm = np.radians(fwhm/60.), verbose=False)
            else:
                cmb_map_smt = cmb_map
            cmb_maps.append(cmb_map_smt)
        return np.array(cmb_maps)
        

def make_cmb_sims(params):
    """ Write cmb maps on disk

    Parameters
    ----------
    params: module contating all the simulation parameters

    """
    instr = getattr(ptep_mbs.instrument, params.inst)
    nmc_cmb = params.nmc_cmb
    nside = params.nside
    smooth = params.gaussian_smooth
    parallel = params.parallel
    start_mc = params.start_mc_num
    root_dir = params.out_dir
    out_dir = f'{root_dir}/cmb/'
    file_str = params.file_string
    channels = instr.keys()
    seed_cmb = params.seed_cmb
    cmb_ps_file = params.cmb_ps_file
    rank = 0
    size = 1
    if params.parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    if not os.path.exists(out_dir) and rank==0:
        os.makedirs(out_dir)
    if cmb_ps_file:
        print(cmb_ps_file)
        cl_cmb = hp.read_cl(cmb_ps_file)
    else:
        cmb_ps_scalar_file = os.path.join(
            os.path.dirname(__file__),
            'datautils/Cls_Planck2018_for_PTEP_2020_r0.fits')
        cl_cmb_scalar = hp.read_cl(cmb_ps_scalar_file)
        cmb_ps_tensor_r1_file = os.path.join(
            os.path.dirname(__file__),
            'datautils/Cls_Planck2018_for_PTEP_2020_tensor_r1.fits')
        cmb_r = params.cmb_r
        cl_cmb_tensor = hp.read_cl(cmb_ps_tensor_r1_file)*cmb_r
        cl_cmb = cl_cmb_scalar+cl_cmb_tensor
    nmc_cmb = math.ceil(nmc_cmb/size)*size
    if nmc_cmb!=params.nmc_cmb:
        print_rnk0(f'WARNING: setting nmc_cmb = {nmc_cmb}', rank)
    perrank = nmc_cmb//size
    for nmc in range(rank*perrank, (rank+1)*perrank):
        if seed_cmb:
            np.random.seed(seed_cmb+nmc)
        nmc_tot = nmc + start_mc
        nmc_str = str(nmc_tot).zfill(4)
        if not os.path.exists(out_dir+nmc_str):
            os.makedirs(out_dir+nmc_str)
        cmb_temp = hp.synfast(cl_cmb, nside, new=True, verbose=False)
        file_name = f'cmb_{nmc_str}_{file_str}.fits'
        file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
        hp.write_map(file_tot_path, cmb_temp, overwrite=True, dtype=np.float32)
        os.environ["PYSM_LOCAL_DATA"] = f'{out_dir}'
        sky = pysm3.Sky(nside=nside, component_objects=[pysm3.CMBMap(nside, map_IQU=f'{nmc_str}/{file_name}')])
        for chnl in channels:
            freq = instr[chnl]['freq']
            if params.band_int:
                band = instr[chnl]['freq_band']
                fmin = freq-band/2.
                fmax = freq+band/2.
                fsteps = fmax-fmin+1
                bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                weights = np.ones(len(bandpass_frequencies))
                cmb_map = sky.get_emission(bandpass_frequencies, weights)
                cmb_map = cmb_map*bandpass_unit_conversion(bandpass_frequencies, weights, u.uK_CMB)
            else:
                cmb_map = sky.get_emission(freq*u.GHz)
                cmb_map = cmb_map.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
            fwhm = instr[chnl]['beam']
            if smooth:
                cmb_map_smt = hp.smoothing(cmb_map, fwhm = np.radians(fwhm/60.), verbose=False)
            else:
                cmb_map_smt = cmb_map
            file_name = f'{chnl}_cmb_{nmc_str}_{file_str}.fits'
            file_tot_path = f'{out_dir}{nmc_str}/{file_name}'
            hp.write_map(file_tot_path, cmb_map_smt, overwrite=True, dtype=np.float32)
