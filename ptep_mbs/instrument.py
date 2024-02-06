import numpy as np

def sens(net):
    sens = np.sqrt(4*np.pi*2.*net**2./(3*365.*24*60*60*0.85*0.95*0.95))*(10800/np.pi)
    return sens

LB_test = {}
LB_test['LB_mock_140'] = {'freq': 140., 'freq_band': 42., 'beam': 30.8, 'P_sens': 6.39}

LB_IMOv1 = {}
LB_IMOv1['LB_LFT_40'] = {'freq': 40., 'freq_band': 12., 'beam': 70.5, 'P_sens': 37.42}
LB_IMOv1['LB_LFT_50'] = {'freq': 50., 'freq_band': 15., 'beam': 58.5, 'P_sens': 33.46}
LB_IMOv1['LB_LFT_60'] = {'freq': 60., 'freq_band': 14., 'beam': 51.1, 'P_sens': 21.31}
LB_IMOv1['LB_LFT_68a'] = {'freq': 68., 'freq_band': 16., 'beam': 41.6, 'P_sens': 19.91}
LB_IMOv1['LB_LFT_68b'] = {'freq': 68., 'freq_band': 16., 'beam': 47.1, 'P_sens': 31.76}
LB_IMOv1['LB_LFT_78a'] = {'freq': 78., 'freq_band': 18., 'beam': 36.9, 'P_sens': 15.56}
LB_IMOv1['LB_LFT_78b'] = {'freq': 78., 'freq_band': 18., 'beam': 43.8, 'P_sens': 19.14}
LB_IMOv1['LB_LFT_89a'] = {'freq': 89., 'freq_band': 20., 'beam': 33.0, 'P_sens': 12.28}
LB_IMOv1['LB_LFT_89b'] = {'freq': 89., 'freq_band': 20., 'beam': 41.5, 'P_sens': 28.77}
LB_IMOv1['LB_LFT_100'] = {'freq': 100., 'freq_band': 23., 'beam': 30.2, 'P_sens': 10.34}
LB_IMOv1['LB_LFT_119'] = {'freq': 119., 'freq_band': 36., 'beam': 26.3, 'P_sens': 7.69}
LB_IMOv1['LB_LFT_140'] = {'freq': 140., 'freq_band': 42., 'beam': 23.7, 'P_sens': 7.24}
LB_IMOv1['LB_MFT_100'] = {'freq': 100., 'freq_band': 23., 'beam': 37.8, 'P_sens': 8.48}
LB_IMOv1['LB_MFT_119'] = {'freq': 119., 'freq_band': 36., 'beam': 33.6, 'P_sens': 5.70}
LB_IMOv1['LB_MFT_140'] = {'freq': 140., 'freq_band': 42., 'beam': 30.8, 'P_sens': 6.39}
LB_IMOv1['LB_MFT_166'] = {'freq': 166., 'freq_band': 50., 'beam': 28.9, 'P_sens': 5.57}
LB_IMOv1['LB_MFT_195'] = {'freq': 195., 'freq_band': 59., 'beam': 28.0, 'P_sens': 7.04}
LB_IMOv1['LB_HFT_195'] = {'freq': 195., 'freq_band': 59., 'beam': 28.6, 'P_sens': 10.50}
LB_IMOv1['LB_HFT_235'] = {'freq': 235., 'freq_band': 71., 'beam': 24.7, 'P_sens': 10.79}
LB_IMOv1['LB_HFT_280'] = {'freq': 280., 'freq_band': 84., 'beam': 22.5, 'P_sens': 13.80}
LB_IMOv1['LB_HFT_337'] = {'freq': 337., 'freq_band': 101., 'beam': 20.9, 'P_sens': 21.95}
LB_IMOv1['LB_HFT_402'] = {'freq': 402., 'freq_band': 92., 'beam': 17.9, 'P_sens': 47.45}

latest = LB_IMOv1

LB_v28 = {}
LB_v28['LB_LFT_40'] = {'freq': 40., 'freq_band': 12., 'beam': 69.3, 'P_sens': 59.29}
LB_v28['LB_LFT_50'] = {'freq': 50., 'freq_band': 15., 'beam': 56.8, 'P_sens': 32.78}
LB_v28['LB_LFT_60'] = {'freq': 60., 'freq_band': 14., 'beam': 49.0, 'P_sens': 25.76}
LB_v28['LB_LFT_68a'] = {'freq': 68., 'freq_band': 16., 'beam': 41.6, 'P_sens': 21.60}
LB_v28['LB_LFT_68b'] = {'freq': 68., 'freq_band': 16., 'beam': 44.5, 'P_sens': 23.53}
LB_v28['LB_LFT_78a'] = {'freq': 78., 'freq_band': 18., 'beam': 36.9, 'P_sens': 18.59}
LB_v28['LB_LFT_78b'] = {'freq': 78., 'freq_band': 18., 'beam': 40.0, 'P_sens': 18.45}
LB_v28['LB_LFT_89a'] = {'freq': 89., 'freq_band': 20., 'beam': 33.0, 'P_sens': 16.95}
LB_v28['LB_LFT_89b'] = {'freq': 89., 'freq_band': 20., 'beam': 36.7, 'P_sens': 15.03}
LB_v28['LB_LFT_100'] = {'freq': 100., 'freq_band': 23., 'beam': 30.2, 'P_sens': 12.93}
LB_v28['LB_LFT_119'] = {'freq': 119., 'freq_band': 36., 'beam': 26.3, 'P_sens': 9.79}
LB_v28['LB_LFT_140'] = {'freq': 140., 'freq_band': 42., 'beam': 23.7, 'P_sens': 9.55}
LB_v28['LB_MFT_100'] = {'freq': 100., 'freq_band': 23., 'beam': 37.8, 'P_sens': 9.67}
LB_v28['LB_MFT_119'] = {'freq': 119., 'freq_band': 36., 'beam': 33.6, 'P_sens': 6.41}
LB_v28['LB_MFT_140'] = {'freq': 140., 'freq_band': 42., 'beam': 30.8, 'P_sens': 7.02}
LB_v28['LB_MFT_166'] = {'freq': 166., 'freq_band': 50., 'beam': 28.9, 'P_sens': 5.81}
LB_v28['LB_MFT_195'] = {'freq': 195., 'freq_band': 59., 'beam': 28.0, 'P_sens': 7.12}
LB_v28['LB_HFT_195'] = {'freq': 195., 'freq_band': 59., 'beam': 28.6, 'P_sens': 15.66}
LB_v28['LB_HFT_235'] = {'freq': 235., 'freq_band': 71., 'beam': 24.7, 'P_sens': 15.16}
LB_v28['LB_HFT_280'] = {'freq': 280., 'freq_band': 84., 'beam': 22.5, 'P_sens': 17.98}
LB_v28['LB_HFT_337'] = {'freq': 337., 'freq_band': 101., 'beam': 20.9, 'P_sens': 24.99}
LB_v28['LB_HFT_402'] = {'freq': 402., 'freq_band': 92., 'beam': 17.9, 'P_sens': 49.90}

LB_hf_1p1 = {}
LB_hf_1p1['LB_HFT_215'] = {'freq': 214.5, 'freq_band': 0., 'beam': 27.7, 'P_sens': 11.98}
LB_hf_1p1['LB_HFT_259'] = {'freq': 258.5, 'freq_band': 0., 'beam': 24.3, 'P_sens': 12.99}
LB_hf_1p1['LB_HFT_308'] = {'freq': 308., 'freq_band': 0., 'beam': 22.0, 'P_sens': 18.54}
LB_hf_1p1['LB_HFT_371'] = {'freq': 370.7, 'freq_band': 0., 'beam': 20.5, 'P_sens': 31.81}
LB_hf_1p1['LB_HFT_442'] = {'freq': 442.2, 'freq_band': 0., 'beam': 17.5, 'P_sens': 74.5}

LB_hf_1p2 = {}
LB_hf_1p2['LB_HFT_234'] = {'freq': 234., 'freq_band': 0., 'beam': 26.0, 'P_sens': 13.45}
LB_hf_1p2['LB_HFT_282'] = {'freq': 282., 'freq_band': 0., 'beam': 23.1, 'P_sens': 16.08}
LB_hf_1p2['LB_HFT_336'] = {'freq': 336., 'freq_band': 0., 'beam': 21.2, 'P_sens': 24.23}
LB_hf_1p2['LB_HFT_404'] = {'freq':404.4, 'freq_band': 0., 'beam': 19.9, 'P_sens': 45.77}
LB_hf_1p2['LB_HFT_482'] = {'freq': 482.4, 'freq_band': 0., 'beam': 17.1, 'P_sens': 120.8}

LB_hf_1p3 = {}
LB_hf_1p3['LB_HFT_254'] = {'freq': 253.5, 'freq_band': 0., 'beam': 24.6, 'P_sens': 15.69} 
LB_hf_1p3['LB_HFT_306'] = {'freq': 305.5, 'freq_band': 0., 'beam': 22.1, 'P_sens': 20.0}
LB_hf_1p3['LB_HFT_364'] = {'freq': 364., 'freq_band': 0., 'beam': 20.6, 'P_sens': 32.74}
LB_hf_1p3['LB_HFT_438'] = {'freq': 438.1, 'freq_band': 0., 'beam': 19.4, 'P_sens': 68.0}
LB_hf_1p3['LB_HFT_523'] = {'freq': 522.6, 'freq_band': 0., 'beam': 16.5, 'P_sens': 202.19}

LB_hf_1p4 = {}
LB_hf_1p4['LB_HFT_273'] = {'freq': 273., 'freq_band': 0., 'beam': 23.5, 'P_sens': 18.42}
LB_hf_1p4['LB_HFT_329'] = {'freq': 329., 'freq_band': 0., 'beam': 21.4, 'P_sens': 25.34}
LB_hf_1p4['LB_HFT_392'] = {'freq': 392., 'freq_band': 0., 'beam': 20.1, 'P_sens': 44.29}
LB_hf_1p4['LB_HFT_472'] = {'freq': 471.8, 'freq_band': 0., 'beam': 18.6, 'P_sens': 100.95}
LB_hf_1p4['LB_HFT_563'] = {'freq': 562.8, 'freq_band': 0., 'beam': 15.6, 'P_sens': 338.75}

LB_hf_1p5 = {}
LB_hf_1p5['LB_HFT_293'] = {'freq': 292.5, 'freq_band': 0., 'beam': 22.6, 'P_sens': 22.01}
LB_hf_1p5['LB_HFT_353'] = {'freq': 352.5, 'freq_band': 0., 'beam': 20.8, 'P_sens': 32.23}
LB_hf_1p5['LB_HFT_420'] = {'freq': 420., 'freq_band': 0., 'beam': 19.7, 'P_sens': 60.65}
LB_hf_1p5['LB_HFT_506'] = {'freq': 505.5, 'freq_band': 0., 'beam': 17.1, 'P_sens': 151.86}
LB_hf_1p5['LB_HFT_603'] = {'freq': 603., 'freq_band': 0., 'beam': 13.9, 'P_sens': 574.55}
