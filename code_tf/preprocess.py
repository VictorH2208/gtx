import scipy.io as sio
import numpy as np

def scale_data(data_dict, params):
    scaled_data_dict = {}
    for key, items in data_dict.items():
        scaled_data_dict[key] = items * params[key]
    return scaled_data_dict

def get_channel_min_max(data):
    """
    Get per-channel min and max values across the entire dataset.

    Args:
        data: numpy array of shape (N, H, W, C)

    Returns:
        list of (min, max) for each channel
    """
    n_channels = data.shape[-1]
    results = []
    for ch in range(n_channels):
        channel_data = data[..., ch]
        channel_data = channel_data[channel_data > 0]
        ch_min = np.min(channel_data)
        ch_max = np.max(channel_data)
        results.append((ch_min, ch_max))
    return results

def minmax_normalize(data, mins_maxs):
    """
    Normalize (N, H, W, C) array channel-wise using given min/max list.
    Keeps zeros as zeros (background).
    """
    out = data.copy()
    C = data.shape[-1]
    for ch in range(C):
        mn, mx = mins_maxs[ch]
        mask = out[..., ch] != 0  # ignore background
        out[..., ch][mask] = (out[..., ch][mask] - mn) / (mx - mn)
    return out

def load_split_data(file_path):
    data = sio.loadmat(file_path)
    
    data = {k: v for k, v in data.items() if not k.startswith('__')}

    splits = ['train', 'val', 'test']
    data_by_split = {split: {} for split in splits}

    for key, value in data.items():
        for split in splits:
            if key.startswith(split + '_'):
                field = key[len(split) + 1:]
                data_by_split[split][field] = value

    return data_by_split

def load_data(file_path, scale_params, normalize=False):
    data_by_split = load_split_data(file_path)

    result = {}
    mins_maxs_fluorescence = None
    mins_maxs_optical_props = None
    for type in ['train', 'val', 'test']:
        fluorescence = data_by_split[type]['fluorescence']
        optical_props = data_by_split[type]['optical_props']
        depth = data_by_split[type]['depth']
        concentration_fluor = data_by_split[type]['concentration_fluor']
        reflectance = data_by_split[type]['reflectance']

        if type == 'train':
            mins_maxs_fluorescence = get_channel_min_max(fluorescence)
            mins_maxs_optical_props = get_channel_min_max(optical_props)
            print("FFFFF:", mins_maxs_fluorescence)
            print("OOOOO:", mins_maxs_optical_props)

        if not normalize:
            scaled_data_dict = scale_data({
                'fluorescence': fluorescence,
                'reflectance': reflectance,
                'depth': depth, 
                'mu_a': optical_props[..., 0],
                'mu_s': optical_props[..., 1],
                'concentration_fluor': concentration_fluor
                }, scale_params)
            
            result[type] = scaled_data_dict
        else:
            normed_op = minmax_normalize(optical_props, mins_maxs_optical_props)
            normalized_data_dict = {
                'fluorescence': minmax_normalize(fluorescence, mins_maxs_fluorescence),
                'reflectance': reflectance,
                'depth': depth, 
                'mu_a': normed_op[..., 0],
                'mu_s': normed_op[..., 1],
                'concentration_fluor': concentration_fluor
            }
            result[type] = normalized_data_dict
            print("Normalize over fluorescence and optical properties")

    return result