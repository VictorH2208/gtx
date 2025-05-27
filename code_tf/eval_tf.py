import os
import sys
import argparse
import numpy as np
import tensorflow as tf

from code_tf.model.model import ModelInit
from utils.preprocess import load_data

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Eval script for fluorescence imaging model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved .keras model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test .mat data')
    
    parser.add_argument('--scaleFL', type=float, default=10e4)
    parser.add_argument('--scaleOP0', type=float, default=10)
    parser.add_argument('--scaleOP1', type=float, default=1)
    parser.add_argument('--scaleDF', type=float, default=1)
    parser.add_argument('--scaleQF', type=float, default=1)
    parser.add_argument('--scaleRE', type=float, default=1)
    parser.add_argument('--batch', type=int, default=32)
    return parser

def eval(params):
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
    fluorescence = np.transpose(fluorescence, (0, 3, 1, 2))
    op = np.stack([data['mu_a'], data['mu_s']], axis=1)
    op = np.transpose(op, (0, 2, 3, 1))
    depth = data['depth']
    concentration_fluor = data['concentration_fluor']
    reflectance = data['reflectance']

    # Load the model
    print(f"Loading model from {params['model_path']}")
    model = tf.keras.models.load_model(params['model_path'])

    # Evaluate
    results = model.evaluate(
        [op, fluorescence],
        {'outQF': concentration_fluor, 'outDF': depth, 'outReflect': reflectance},
        batch_size=params['batch'],
        verbose=1
    )

    print("\nEvaluation results:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # Optionally: save predictions for later analysis
    # predictions = model.predict([op, fluorescence], batch_size=params['batch'])
    # np.savez("eval_preds.npz", **predictions)

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)

    np.random.seed(1024)
    tf.random.set_seed(1024)

    eval(params)