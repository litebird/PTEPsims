##### general configuration
parallel = False

##### define LiteBIRD instrument
inst = 'LB_v28'

##### output maps parameters
nside = 512
gaussian_smooth = True
save_coadd = False

##### noise configuration
make_noise = True
nmc_noise = 1
seed_noise = 9876
N_split = 3

##### cmb configuration
make_cmb = True
cmb_ps_file = False
cmb_r = 0.01
nmc_cmb = 1
seed_cmb = 1234

##### foregrund configuration
make_fg = True
fg_dir = '/marconi_work/INF24_litebird/anto/fgs'
fg_models = {
    "dust": 'pysm_dust_0.cfg',
    }

##### output options
out_dir = 'test'
file_string = 'test_v0'
