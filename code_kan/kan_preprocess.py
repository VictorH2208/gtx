import scipy.io as sio
import mat73
import numpy as np
import os
os.chdir('/home/victorhuang/projects/gtx/')

def load_data(file_path, data_type='dt'):

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

    if data_type == 'dt':
        data = load_split_data(file_path)
        train_data = data['train']
        val_data = data['val']
        test_data = data['test']

        train_fluorescence = train_data['fluorescence'].transpose(0, 3, 1, 2)
        train_mua = train_data['optical_props'][..., 0] + 0.033
        train_mus = train_data['optical_props'][..., 1]
        train_op = np.stack([train_mua, train_mus], axis=1)
        train_depth = train_data['depth']
        train_concentration_fluor = train_data['concentration_fluor']
        train_reflectance = train_data['reflectance']

        val_fluorescence = val_data['fluorescence'].transpose(0, 3, 1, 2)
        val_mua = val_data['optical_props'][..., 0] + 0.033
        val_mus = val_data['optical_props'][..., 1]
        val_op = np.stack([val_mua, val_mus], axis=1)
        val_depth = val_data['depth']
        val_concentration_fluor = val_data['concentration_fluor']
        val_reflectance = val_data['reflectance']

        test_fluorescence = test_data['fluorescence'].transpose(0, 3, 1, 2)
        test_mua = test_data['optical_props'][..., 0] + 0.033
        test_mus = test_data['optical_props'][..., 1]
        test_op = np.stack([test_mua, test_mus], axis=1)
        test_depth = test_data['depth']
        test_concentration_fluor = test_data['concentration_fluor']
        test_reflectance = test_data['reflectance']
            
        data = {
            'train': {
                'fluorescence': train_fluorescence,
                'op': train_op,
                'depth': train_depth,
                'concentration_fluor': train_concentration_fluor,
                'reflectance': train_reflectance
            },
            'val': {
                'fluorescence': val_fluorescence,
                'op': val_op,
                'depth': val_depth,
                'concentration_fluor': val_concentration_fluor,
                'reflectance': val_reflectance
            },
            'test': {
                'fluorescence': test_fluorescence,
                'op': test_op,
                'depth': test_depth,
                'concentration_fluor': test_concentration_fluor,
                'reflectance': test_reflectance
            }
        }

    elif data_type == 'phantoms':
        data = mat73.loadmat(file_path)
        fluorescence = data['F']
        optical_props = data['OP']
        optical_props[..., 0] += 0.033
        reflectance = data['RE']
        depth = data['DF']
        concentration = data['QF']

        data = {
            'fluorescence': fluorescence,
            'op': optical_props,
            'depth': depth,
            'concentration': concentration,
            'reflectance': reflectance
        } 

    return data