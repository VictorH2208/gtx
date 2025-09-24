import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERRORs

# project_root = '/home/victorh/projects/gtx'
# os.chdir(project_root)
# sys.path.insert(0, project_root)

import logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

import argparse
import numpy as np
from model.model_shape import ModelInit
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
import keras
from keras import callbacks, optimizers
from preprocess import load_data

np.random.seed(1024)
tf.random.set_seed(1024)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        print(gpus)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

def get_arg_parser():
    parser = argparse.ArgumentParser(description="Hyperparameter configuration for fluorescence imaging model.")

    parser.add_argument('--sagemaker', type=bool, default=False, help='SageMaker mode')
    parser.add_argument('--model_name', type=str, default='model_hikaru', help='Model name')
    parser.add_argument('--train_subset', type=int, default=8000, help='Train subset')
    parser.add_argument('--seed', type=int, default=1024, help='Seed')

    # General hyperparameters
    parser.add_argument('--activation', type=str, default='relu', help='Activation function')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer name')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--nF', type=int, default=6, help='Number of fluroescent spatial frequencies (fluorescent images)')
    parser.add_argument('--learningRate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--xX', type=int, default=101, help='Image width')
    parser.add_argument('--yY', type=int, default=101, help='Image height')
    parser.add_argument('--decayRate', type=float, default=0.3, help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--normalize', type=int, default=0, help='Normalize data')
    parser.add_argument('--depth_padding', type=int, default=0, help='Depth padding')
    parser.add_argument('--fx_idx', type=int, nargs=6, default=[0, 1, 2, 3, 4, 5])
    # Scaling parameters
    parser.add_argument('--scaleFL', type=float, default=10e4, help='Scaling factor for fluorescence')
    parser.add_argument('--scaleOP0', type=float, default=10, help='Scaling for absorption coefficient (μa)')
    parser.add_argument('--scaleOP1', type=float, default=1, help='Scaling for scattering coefficient (μs)')
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
    parser.add_argument('--data_path', type=str, default='../data/ts_2d_10000.mat')
    parser.add_argument('--model_dir', type=str, default='../code_tf/aws_ckpt/')
    return parser
    
class BatchLogger(callbacks.Callback):
    def __init__(self, log_interval=50, num_samples=None, batch_size=None):
        super().__init__()
        self.log_interval = log_interval
        self.total_batches = num_samples // batch_size

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.log_interval == 0:
            loss = logs.get('loss')
            logging.info(f"[Batch {batch}/{self.total_batches}] Loss: {loss:.4f}")

    def on_epoch_end(self, epoch, logs=None):
        logging.info(
            f"Epoch {epoch}: "
            + ", ".join([f"{k} = {v:.4f}" for k, v in logs.items() if isinstance(v, float)])
        )

class CustomModelCheckpoint(callbacks.Callback):
    def __init__(self, filepath, model_wrapper, monitor='val_loss', verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.best = float('inf')
        self.verbose = verbose
        self.model_wrapper = model_wrapper

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is not None and current < self.best:
            if self.verbose:
                print(f"Epoch {epoch}: {self.monitor} improved from {self.best:.5f} to {current:.5f}, saving models...")
            self.best = current

            # legacy_path = os.path.join(self.filepath, 'model_ckpt_tf')
            # self.model.save(legacy_path, save_format='tf')

            # export_path = os.path.join(self.filepath, 'model_ckpt')
            # self.model.export(export_path)
            path = os.path.join(self.filepath, 'model.keras')
            self.model_wrapper.save_model(path)

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

    if params['sagemaker']:
        filepath = os.path.join('/opt/ml/input/data/training', params['data_path'])
        data = load_data(filepath, scale_params, params['seed'], params['normalize'])
    else:
        filepath = os.path.join('../data', params['data_path'])
        data = load_data(filepath, scale_params, params['seed'], params['normalize'])

    train_data = data['train'].copy()
    train_fluorescence = train_data['fluorescence']
    train_fluorescence = train_fluorescence[...,params['fx_idx']]
    # train_fluorescence = np.transpose(train_fluorescence, (0, 3, 1, 2))
    train_fluorescence = np.expand_dims(train_fluorescence, axis=-1)
    train_op = np.stack([train_data['mu_a'], train_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    train_depth = train_data['depth']
    train_shape = train_data['depth'].copy()
    train_shape[train_shape != 0] = 1

    N = train_fluorescence.shape[0]
    if params['train_subset'] and 0 < params['train_subset'] < N:
        rng = np.random.RandomState(1024)
        idx = rng.choice(N, size=params['train_subset'], replace=False)
        train_fluorescence       = train_fluorescence[idx]
        train_op                  = train_op[idx]
        train_depth               = train_depth[idx]
        train_shape               = train_shape[idx]

    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_op, train_fluorescence),  # tuple of inputs
        {'outDF': train_depth, 'outShape': train_shape}  # dict of outputs
    ))
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=1024, reshuffle_each_iteration=False).batch(params['batch'])

    val_data = data['val'].copy()
    val_fluorescence = val_data['fluorescence']
    val_fluorescence = val_fluorescence[...,params['fx_idx']]
    val_fluorescence = np.expand_dims(val_fluorescence, axis=-1)
    val_op = np.stack([val_data['mu_a'], val_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    val_depth = val_data['depth']
    val_shape = val_data['depth'].copy()
    val_shape[val_shape != 0] = 1

    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_op, val_fluorescence),  # tuple of inputs
        {'outDF': val_depth, 'outShape': val_shape}  # dict of outputs
    ))
    val_dataset = val_dataset.batch(params['batch'])

    test_data = data['test'].copy()
    test_fluorescence = test_data['fluorescence']
    test_fluorescence = test_fluorescence[...,params['fx_idx']]
    test_fluorescence = np.expand_dims(test_fluorescence, axis=-1)
    test_op = np.stack([test_data['mu_a'], test_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    test_depth = test_data['depth']
    test_shape = test_data['depth'].copy()
    test_shape[test_shape != 0] = 1

    test_dataset = tf.data.Dataset.from_tensor_slices((
        (test_op, test_fluorescence),  # tuple of inputs
        {'outDF': test_depth, 'outShape': test_shape}  # dict of outputs
    ))
    test_dataset = test_dataset.batch(params['batch'])

    print("Train dataset shape:", train_dataset.element_spec)
    print("Val dataset shape:", val_dataset.element_spec)
    print("Test dataset shape:", test_dataset.element_spec)

    # Initialize model
    model = ModelInit(params)
    model.build_model()

    # Initialize optimizer
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=params['decayRate'], patience=5, verbose=1, min_lr=5e-5)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=5e-5, patience=params['patience'], verbose=1, mode='auto')
    batch_logger = BatchLogger(log_interval=20, num_samples= int(train_fluorescence.shape[0]), batch_size=params['batch'])

    if params['sagemaker']:
        model_dir = f'/opt/ml/model'
    else:
        model_dir = os.path.join(params['model_dir'], f"subset{params['train_subset']}-epochs{params['epochs']}-batch{params['batch']}-data{params['data_path'].split('.')[0]}")
        os.makedirs(model_dir, exist_ok=True)

    checkpoint = CustomModelCheckpoint(
        filepath=model_dir,
        model_wrapper=model,
        monitor='val_loss',
        verbose=1
    )
    csv_logger = callbacks.CSVLogger(os.path.join(model_dir, f'loss.log'), append=True)
    cb = [lr_scheduler, early_stopping, batch_logger, checkpoint, csv_logger]
    
    print("++++++++ Keras version: ", keras.__version__, keras.__file__)

    # Train model
    model.model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=params['epochs'],
        verbose=0,  
        callbacks=cb
    )
    

    model.load_model(os.path.join(model_dir, f'model.keras'))
    model.model.evaluate(
        test_dataset,
        verbose=1
    )

        
if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)

    np.random.seed(1024)
    tf.random.set_seed(1024)

    train(params)





