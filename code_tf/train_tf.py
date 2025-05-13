import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERRORs

project_root = '/home/victorh/projects/gtx'
os.chdir(project_root)
sys.path.insert(0, project_root)

import argparse
import numpy as np
from model import ModelInit
from tqdm import tqdm
import tensorflow as tf

from utils.preprocess import load_data

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for fluorescence imaging model.")

    # General hyperparameters
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--nF', type=int, default=6, help='Number of fluroescent spatial frequencies (fluorescent images)')
    parser.add_argument('--learningRate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--xX', type=int, default=101, help='Image width')
    parser.add_argument('--yY', type=int, default=101, help='Image height')
    parser.add_argument('--decayRate', type=float, default=0.3, help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')

    # Scaling parameters
    parser.add_argument('--scaleFL', type=float, default=10e4, help='Scaling factor for fluorescence')
    parser.add_argument('--scaleOP0', type=float, default=10, help='Scaling for absorption coefficient (μa)')
    parser.add_argument('--scaleOP1', type=float, default=1, help='Scaling for scattering coefficient (μs\')')
    parser.add_argument('--scaleDF', type=float, default=1, help='Scaling for depth')
    parser.add_argument('--scaleQF', type=float, default=1, help='Scaling for fluorophore concentration')
    parser.add_argument('--scaleRE', type=float, default=1, help='Scaling for reflectance (optional)')

    # 3D Conv parameters
    parser.add_argument('--nFilters3D', type=int, default=128)
    parser.add_argument('--kernelConv3D', type=int, nargs=3, default=[3,3,3])
    parser.add_argument('--strideConv3D', type=int, nargs=3, default=[1,1,1])

    # 2D Conv parameters
    parser.add_argument('--nFilters2D', type=int, default=128)
    parser.add_argument('--kernelConv2D', type=int, nargs=2, default=[3,3])
    parser.add_argument('--strideConv2D', type=int, nargs=2, default=[1,1])

    # Data path
    parser.add_argument('--data_path', type=str, default='data/')

    return parser

def train(params):

    # Load data
    scale_params = {
        'fluorescence': params['scaleFL'],
        'mu_a': params['scaleOP0'],
        'mu_s': params['scaleOP1'],
        'depth': params['scaleDF'],
        'concentration_fluor': params['scaleQF'],
        'reflectance': params['scaleRE']
    }
    data = load_data(params['data_path'], scale_params)

    fluorescence = data['fluorescence']
    op = np.stack([data['mu_a'], data['mu_s']], axis=1)
    op = np.transpose(op, (0, 2, 3, 1))
    depth = data['depth']
    concentration_fluor = data['concentration_fluor']

    print(op.shape)
    print(fluorescence.shape)
    print(depth.shape)
    print(concentration_fluor.shape)
    
    return

    # Initialize model
    model = ModelInit(params)
    model.build_model()

    # Initialize optimizer
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=params['decayRate'], patience=5, verbose=1, min_lr=5e-5)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=params['patience'], verbose=1, mode='auto')
    callbacks = [lr_scheduler, early_stopping]

    # Train model
    model.model.fit([op, fluorescence], [concentration_fluor, depth], validation_split=0.2, batch_size=params['batch'], epochs=params['epochs'], verbose=1, shuffle=True, callbacks=callbacks)
        
        
if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)

    train(params)





