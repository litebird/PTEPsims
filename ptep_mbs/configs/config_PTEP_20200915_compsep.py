#general configuration
parallel = True

#define LiteBIRD instrument
inst = 'LB_IMOv1'

#output maps parameters
nside = 512
gaussian_smooth = True
band_int = False
save_coadd = True

#noise configuration
make_noise = True
nmc_noise = 100
seed_noise = 6437
hm_split = True

# #cmb configuration
make_cmb = True
#cmb_ps_file = 'Cls_Planck2018_for_PTEP_2020_r0.fits'
cmb_r = 0
nmc_cmb = 100
seed_cmb = 38198

# #foregrund configuration
make_fg = True
fg_dir = '/marconi_work/INF24_litebird/anto/fgs'
fg_models = {
    "dust": 'pysm_dust_1.cfg',
    "synch": 'pysm_synch_1.cfg',
    "ame": 'pysm_ame_1.cfg',
    "freefree": 'pysm_freefree_1.cfg',
    }

#output options
out_dir = './'
file_string = 'PTEP_20200915_compsep'

