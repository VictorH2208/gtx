import scipy.io as sio
import mat73
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

def split_indices(n_samples, val_size=0.1, test_size=0.1, random_state=1024):
    all_idx = np.arange(n_samples)
    np.random.seed(random_state)
    np.random.shuffle(all_idx)

    n_val = int(n_samples * val_size)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_val - n_test

    train_idx = all_idx[:n_train]
    val_idx = all_idx[n_train:n_train + n_val]
    test_idx = all_idx[n_train + n_val:]
    
    return train_idx, val_idx, test_idx

def load_split_data(file_path, seed):


    try:
        data = mat73.loadmat(file_path)
    except Exception as e1:
        try:
            data = sio.loadmat(file_path)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load {file_path} with both mat73 and scipy.io.\n"
                f"mat73 error: {e1}\n"
                f"scipy.io error: {e2}"
            )

    fluorescence = data['F']    
    reflectance = data['RE']  
    depth = data['DF']          
    optical_props = data['OP']  
    concentration_fluor = data['QF']  

    n_samples = data['F'].shape[0]
    train_idx, val_idx, test_idx = split_indices(n_samples, random_state=seed)

    split_data = {
        'train_fluorescence': fluorescence[train_idx],
        'val_fluorescence': fluorescence[val_idx],
        'test_fluorescence': fluorescence[test_idx],

        'train_optical_props': optical_props[train_idx],
        'val_optical_props': optical_props[val_idx],
        'test_optical_props': optical_props[test_idx],

        'train_reflectance': reflectance[train_idx],
        'val_reflectance': reflectance[val_idx],
        'test_reflectance': reflectance[test_idx],

        'train_depth': depth[train_idx],
        'val_depth': depth[val_idx],
        'test_depth': depth[test_idx],

        'train_concentration_fluor': concentration_fluor[train_idx],
        'val_concentration_fluor': concentration_fluor[val_idx],
        'test_concentration_fluor': concentration_fluor[test_idx],
    }
    
    data = {k: v for k, v in split_data.items() if not k.startswith('__')}

    splits = ['train', 'val', 'test']
    data_by_split = {split: {} for split in splits}

    for key, value in data.items():
        for split in splits:
            if key.startswith(split + '_'):
                field = key[len(split) + 1:]
                data_by_split[split][field] = value

    return data_by_split

def load_data(file_path, scale_params, seed, normalize=False):
    data_by_split = load_split_data(file_path, seed)

    result = {}
    mins_maxs_fluorescence = None
    mins_maxs_optical_props = None
    for type in ['train', 'val', 'test']:
        fluorescence = data_by_split[type]['fluorescence']
        optical_props = data_by_split[type]['optical_props']
        depth = data_by_split[type]['depth']
        concentration_fluor = data_by_split[type]['concentration_fluor']
        reflectance = data_by_split[type]['reflectance']

        fr = fluorescence / (reflectance + 1e-20)

        if type == 'train':
            mins_maxs_fr = get_channel_min_max(fr)
            mins_maxs_optical_props = get_channel_min_max(optical_props)
            print("FFFFF:", mins_maxs_fr)
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
            normed_fr = minmax_normalize(fr, mins_maxs_fr)
            normalized_data_dict = {
                'fluorescence': normed_fr,
                'reflectance': reflectance,
                'depth': depth, 
                'mu_a': normed_op[..., 0],
                'mu_s': normed_op[..., 1],
                'concentration_fluor': concentration_fluor
            }
            result[type] = normalized_data_dict
            print("Normalize over fluorescence and optical properties")

    return result