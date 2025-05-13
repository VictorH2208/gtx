import mat73
import numpy as np

def scale_data(data_dict, params):
    scaled_data_dict = {}
    for key, items in data_dict.items():
        scaled_data_dict[key] = items * params[key]
    return scaled_data_dict

def normalization(fluorescence, optical_props):
    f = (fluorescence - np.mean(fluorescence, axis=(1,2,3), keepdims=True)) / \
                   (np.std(fluorescence, axis=(1,2,3), keepdims=True) + 1e-6)
    mu_a = optical_props[..., 0]
    mu_s = optical_props[..., 1]

    mu_a_mean = np.mean(mu_a, axis=(1,2), keepdims=True)
    mu_a_std = np.std(mu_a, axis=(1,2), keepdims=True)
    mu_a_norm = (mu_a - mu_a_mean) / (mu_a_std + 1e-6)

    mu_s_mean = np.mean(mu_s, axis=(1,2), keepdims=True)
    mu_s_std = np.std(mu_s, axis=(1,2), keepdims=True)
    mu_s_norm = (mu_s - mu_s_mean) / (mu_s_std + 1e-6)
    return f, mu_a_norm, mu_s_norm

def load_data(file_path, scale_params):
    data = mat73.loadmat(file_path)
    fluorescence = data['F']    # FL → fluorescence
    reflectance = data['RE']  # RE → radiance of the sample
    depth = data['DF']          # DF → depth
    optical_props = data['OP']  # OP → optical properties (μa, μs')
    concentration_fluor = data['QF']  # QF → corrected fluorophore concentration

    
    f, mu_a_norm, mu_s_norm = normalization(fluorescence, optical_props)
    scaled_data_dict = scale_data({
        'fluorescence': f,
        'reflectance': reflectance,
        'depth': depth, 
        'mu_a': mu_a_norm,
        'mu_s': mu_s_norm,
        'concentration_fluor': concentration_fluor
        }, scale_params)
    print("Data loaded and scaled")
    return scaled_data_dict