import os
os.chdir('/home/victorhuang/projects/gtx')
# os.chdir('/home/victorh/projects/gtx')

import numpy as np
import mat73
import json
import scipy.io as sio
import matplotlib.pyplot as plt

# Spatial frequency
fx = [0, 0.05, 0.1, 0.15, 0.2, 0.25] 
fx = np.array(fx)

dt_data_path = 'data/20241118_data_splited.mat'
phantom_data_path = 'data/phantom_data.mat'
cylinder_data_path_r1 = 'data/cylinder/h1_r1_d2.mat'
cylinder_data_path_r2 = 'data/cylinder/h1_r2_d2.mat'  

def opt_prop_india_ink(lambda_nm, pct_india_ink):
    """
    Calculate absorption coefficient (mua) of India ink at given wavelengths.

    Parameters:
    - lambda_nm: array-like, wavelengths in nm
    - pct_india_ink: float, India ink concentration in percent
    - spectra: str, name of the spectra dataset (currently only one supported)

    Returns:
    - mua: ndarray, absorption coefficient in mm^-1
    """
    def get_india_ink(lambda_nm, spectra):
        Wavelength = np.arange(380, 901)

        if spectra == '20140318_Sample2_Sean':
            muaSpectrum = 1 / 10 * 100 * np.array([1.233,1.22929,1.225731,1.222326,1.219081,1.216,1.213087,1.210347,1.207784,1.205404,1.203209,1.201192,1.199269,1.197335,1.195281,1.193,1.19043,1.187686,1.184927,1.182312,1.18,1.178098,1.176503,1.175059,1.17361,1.172,1.170113,1.16799,1.165711,1.163354,1.161,1.158714,1.1565,1.154351,1.152255,1.150203,1.148185,1.146179,1.144162,1.14211,1.14,1.137819,1.135602,1.133396,1.131245,1.129197,1.127279,1.125413,1.123486,1.121386,1.119,1.116265,1.113315,1.110332,1.1075,1.105,1.102951,1.101211,1.099572,1.097829,1.095775,1.093286,1.09049,1.087561,1.084673,1.082,1.079651,1.077484,1.075291,1.072865,1.07,1.066567,1.062753,1.058825,1.055049,1.051691,1.048902,1.046492,1.044208,1.041795,1.039,1.035646,1.031863,1.027857,1.023834,1.02,1.016515,1.013348,1.010424,1.007667,1.005,1.002369,0.999812,0.997388,0.995158,0.993181,0.991504,0.990095,0.988897,0.987854,0.986908,0.986007,0.985107,0.984167,0.983145,0.982,0.980707,0.979306,0.97785,0.976397,0.975,0.973693,0.972419,0.971099,0.969652,0.968,0.966088,0.963962,0.961691,0.959347,0.957,0.954699,0.952407,0.950065,0.947616,0.945,0.942181,0.939204,0.936137,0.933047,0.93,0.927047,0.924174,0.921349,0.91854,0.915716,0.912847,0.909903,0.906857,0.90368,0.900343,0.896843,0.893297,0.889867,0.886714,0.884,0.881832,0.880106,0.878664,0.877348,0.876,0.8745,0.872881,0.871211,0.869561,0.868,0.866581,0.865292,0.8641,0.862977,0.861892,0.860825,0.859787,0.858792,0.857858,0.857,0.856223,0.855486,0.854738,0.853926,0.853,0.851933,0.850802,0.84971,0.848759,0.848053,0.847664,0.847506,0.847436,0.847315,0.847,0.846394,0.845568,0.844639,0.843721,0.84293,0.842331,0.841846,0.841366,0.840787,0.84,0.838939,0.83769,0.836381,0.835139,0.834089,0.833328,0.832775,0.832294,0.831748,0.831,0.829957,0.828706,0.827375,0.826097,0.825,0.824177,0.823562,0.823052,0.822543,0.821931,0.821146,0.820215,0.819184,0.818097,0.817,0.815931,0.814897,0.813898,0.812932,0.812,0.811102,0.810244,0.809436,0.808685,0.808,0.807374,0.806741,0.806022,0.805135,0.804,0.802579,0.801003,0.799444,0.798077,0.797074,0.796561,0.796414,0.796421,0.796371,0.796054,0.795304,0.794187,0.792853,0.791448,0.790123,0.788994,0.788022,0.787114,0.786176,0.785116,0.783867,0.782506,0.781157,0.779947,0.779,0.778397,0.778039,0.777783,0.777484,0.777,0.776228,0.775234,0.774126,0.773012,0.772,0.771162,0.770433,0.769712,0.768897,0.767887,0.766639,0.765273,0.763942,0.7628,0.762,0.761634,0.761557,0.761563,0.761446,0.761,0.760095,0.758897,0.757648,0.75659,0.755964,0.755903,0.756222,0.756675,0.757016,0.757,0.756452,0.755484,0.754281,0.753024,0.751898,0.75102,0.750309,0.749647,0.748916,0.748,0.746818,0.745436,0.743958,0.742486,0.741126,0.739945,0.738824,0.737581,0.736034,0.734,0.731368,0.72831,0.725068,0.721884,0.719,0.716582,0.714489,0.712506,0.710415,0.708,0.705114,0.701893,0.698543,0.695268,0.692276,0.68973,0.687577,0.685687,0.683931,0.68218,0.680335,0.678464,0.676686,0.675123,0.673896,0.673087,0.672657,0.672546,0.672694,0.673042,0.673531,0.674114,0.674742,0.675369,0.675947,0.676444,0.676916,0.677444,0.678111,0.679,0.680156,0.681476,0.682817,0.684039,0.685,0.685591,0.685832,0.685779,0.685484,0.685,0.684393,0.683772,0.683256,0.682967,0.683025,0.683481,0.684191,0.68497,0.685634,0.686,0.685941,0.685556,0.685,0.68443,0.684,0.683823,0.683838,0.683942,0.684031,0.684,0.683775,0.683392,0.682918,0.682418,0.681958,0.681567,0.681172,0.680675,0.679983,0.679,0.677684,0.676204,0.674782,0.67364,0.673,0.672988,0.673347,0.673724,0.673765,0.673116,0.671529,0.669317,0.66698,0.665023,0.663946,0.664039,0.664956,0.666234,0.66741,0.668023,0.667765,0.666792,0.665346,0.663668,0.662,0.660583,0.65966,0.659471,0.660258,0.662262,0.660154,0.659611,0.659069,0.658526,0.657984,0.657441,0.656899,0.656356,0.655814,0.655271,0.654729,0.654187,0.653644,0.653102,0.652559,0.652017,0.651474,0.650932,0.650389,0.649847,0.649304,0.648762,0.648219,0.647677,0.647134,0.646592,0.646049,0.645507,0.644965,0.644422,0.64388,0.643337,0.642795,0.642252,0.64171,0.641167,0.640625,0.640082,0.63954,0.638997,0.638455,0.637912,0.63737,0.636827,0.636285,0.635743,0.6352,0.634658,0.634115,0.633573,0.63303,0.632488,0.631945,0.631403,0.63086,0.630318,0.629775,0.629233,0.62869,0.628148,0.627606,0.627063,0.626521,0.625978,0.625436,0.624893,0.624351,0.623808,0.623266,0.622723,0.622181,0.621638,0.621096,0.620553,0.620011,0.619468,0.618926,0.618384,0.617841,0.617299,0.616756,0.616214,0.615671,0.615129,0.614586,0.614044,0.613501,0.612959,0.612416,0.611874,0.611331,0.610789,0.610246,0.609704,0.609162,0.608619,0.608077,0.607534,0.606992,0.606449])  # Truncated for brevity

        else:
            raise ValueError(f"Spectra '{spectra}' not supported.")


        return np.interp(lambda_nm, Wavelength, muaSpectrum)

    # Get stock mua values at desired wavelengths
    mua_stock = get_india_ink(np.array(lambda_nm), "20140318_Sample2_Sean")

    # Scale by the actual percentage of India Ink
    mua = (pct_india_ink / 100.0) * mua_stock
    return mua


def opt_prop_hb(lambda_vals, fHb, StO2):
    """
    Blood absorption spectrum (oxy- and deoxy-hemoglobin).
    
    Parameters:
        lambda_vals (array-like): Wavelengths (in nm)
        fHb (float): Hemoglobin concentration (g/L)
        StO2 (float): Oxygen saturation (fractional, e.g., 0.95)
    
    Returns:
        mua (np.ndarray): Absorption coefficients (mm^-1)
    """
    
    def getHb(lambda_vals):
        """
        Get absorption coefficients for oxy- and deoxy-hemoglobin.
        Source: http://omlc.org/spectra/hemoglobin/
        
        Parameters:
            lambda_vals (array-like): Wavelengths (in nm)
        
        Returns:
            muaHbO2, muaHb (np.ndarray, np.ndarray): Absorption coefficients in mm^-1/(g/L)
        """
        # Fill in the actual lists with full data or load from a file
        Wavelength = np.arange(250, 1301)  # List of wavelengths (nm)
        with open("numerical/hbo2.json", "r") as f:
            HbO2 = json.load(f)
        with open("numerical/hb.json", "r") as f:
            Hb = json.load(f)         
        
        HbO2 = np.array(HbO2)
        Hb = np.array(Hb)

        # Convert to mm^-1/(g/L)
        HbO2 = HbO2 / 10.0
        Hb = Hb / 10.0

        # Interpolation
        muaHbO2 = np.interp(lambda_vals, Wavelength, HbO2)
        muaHb = np.interp(lambda_vals, Wavelength, Hb)

        return muaHbO2, muaHb
    
    muaHbO2, muaHb = getHb(lambda_vals)
    mua = fHb * (StO2 * muaHbO2 + (1 - StO2) * muaHb)
    return mua

def opt_prop_intralipid(lambda_nm, pctIL20, background='Water', method='Mie'):
    """
    Compute reduced scattering coefficient (musp), scattering coefficient (mu_s), 
    and anisotropy (g) for Intralipid based on the provided wavelength and concentration.

    Parameters:
        lambda_nm (float or np.ndarray): Wavelength in nm
        pctIL20 (float): Percentage of 20% Intralipid stock solution
        background (str): Medium in which intralipid is diluted ('Water', 'Agar', 'Gelatin')
        method (str): Scattering model ('Mie' or 'Flock')

    Returns:
        musp (np.ndarray): Reduced scattering coefficient (1/mm)
        mu_s (np.ndarray): Scattering coefficient (1/mm)
        g (np.ndarray): Anisotropy factor
    """

    lambda_nm = np.asarray(lambda_nm)

    if method == 'Mie':
        # van Staveren et al. (1991)
        mu_s = 2.54e9 * lambda_nm**(-2.4)
        g = 1.1 - 0.58e-3 * lambda_nm
    elif method == 'Flock':
        # Flock et al. (1992)
        mu_s = 1.17e9 * lambda_nm**(-2.33)
        g = 2.25 * lambda_nm**(-0.155)
    else:
        raise ValueError('Unsupported method. Choose "Mie" or "Flock".')

    musp_20pct = 2 * mu_s * (1 - g) / 10
    musp = (pctIL20 / 100) * musp_20pct

    # Background correction
    if 'Agar' in background:
        mus_scale = 0.7
    elif 'Gelatin' in background:
        mus_scale = 0.5
    elif 'Water' in background:
        mus_scale = 1.0
    else:
        raise ValueError('Incorrect background medium specified.')

    musp *= mus_scale

    return musp, mu_s, g


def extrapolate_opt_prop(mua_x, mus_x, lambda_x, lambda_m, absorber, scatterer):
    """
    Extrapolate optical properties (mua, mus) from excitation (lambda_x) to emission (lambda_m) wavelength.

    Parameters:
        mua_x (float or np.ndarray): mua at excitation wavelength
        mus_x (float or np.ndarray): mus at excitation wavelength
        lambda_x (float): excitation wavelength (nm)
        lambda_m (float): emission wavelength (nm)
        absorber (str): 'IndiaInk' or 'Hb'
        scatterer (str): 'Intralipid'

    Returns:
        mua_m (float or np.ndarray): mua at emission wavelength
        mus_m (float or np.ndarray): mus at emission wavelength
    """
    # --- Absorber scaling ---
    if absorber == 'IndiaInk':
        pct_india_ink = 100
        mua_abs_x = opt_prop_india_ink(lambda_x, pct_india_ink)
        mua_abs_m = opt_prop_india_ink(lambda_m, pct_india_ink)
    elif absorber == 'Hb':
        fHb = 1.5
        StO2 = 0.95
        mua_abs_x = opt_prop_hb(lambda_x, fHb, StO2)
        mua_abs_m = opt_prop_hb(lambda_m, fHb, StO2)
    else:
        raise ValueError('Unsupported absorber type')

    mua_scale_emission = mua_abs_m / mua_abs_x
    mua_m = mua_scale_emission * mua_x

    # --- Scatterer scaling ---
    if scatterer == 'Intralipid':
        pct_il = 100
        _, mus_x_ref, _ = opt_prop_intralipid(lambda_x, pct_il)
        _, mus_m_ref, _ = opt_prop_intralipid(lambda_m, pct_il)
    else:
        raise ValueError('Unsupported scatterer type')

    mus_scale_emission = mus_m_ref / mus_x_ref
    mus_m = mus_scale_emission * mus_x

    return mua_m, mus_m