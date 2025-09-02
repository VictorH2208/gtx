import os
import sys

os.environ["KERAS_BACKEND"] = "tensorflow" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = filter INFO, 2 = filter INFO+WARNING, 3 = only ERRORs

import logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
import keras
from keras import callbacks, optimizers, metrics
from keras.saving import register_keras_serializable

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

@register_keras_serializable(package="custom")
def tumor_mae(y_true, y_pred):
    mask = tf.not_equal(y_true, 0.0)
    err  = tf.abs(y_true - y_pred)
    masked_err = tf.boolean_mask(err, mask)
    return tf.cond(tf.size(masked_err) > 0,
                   lambda: tf.reduce_mean(masked_err),
                   lambda: tf.constant(0.0, dtype=y_true.dtype))

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

            path = os.path.join(self.filepath, 'model.keras')
            self.model_wrapper.save(path)


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--is_aws', type=bool, default=False)
    parser.add_argument('--names_to_train', type=str, required=True)
    parser.add_argument('--batch', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--learningRate', type=float, required=True)
    parser.add_argument('--decayRate', type=float, required=True)
    parser.add_argument('--patience', type=int, required=True)

    parser.add_argument('--scaleFL', type=float, required=True)
    parser.add_argument('--scaleOP0', type=float, required=True)
    parser.add_argument('--scaleOP1', type=float, required=True)
    parser.add_argument('--scaleDF', type=float, required=True)
    parser.add_argument('--scaleQF', type=float, required=True)
    parser.add_argument('--scaleRE', type=float, required=True)
    
    return parser

def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

def transfer_learning(params):

    model_path = params['model_path']
    data_path = params['data_path']
    names_to_train = params['names_to_train']
    is_aws = params['is_aws']
    batch = params['batch']
    epochs = params['epochs']
    learning_rate = params['learningRate']
    decay_rate = params['decayRate']
    patience = params['patience']
    scale_params = {
        'fluorescence': params['scaleFL'],
        'mu_a': params['scaleOP0'],
        'mu_s': params['scaleOP1'],
        'depth': params['scaleDF'],
        'concentration_fluor': params['scaleQF'],
        'reflectance': params['scaleRE']
    }

    if is_aws:
        filepath = os.path.join('/opt/ml/input/data/training', data_path)
        model_ckpt = os.path.join('/opt/ml/input/data/ckpt', model_path)
        data = load_data(filepath, scale_params)
    else:
        data_path = os.path.join('../data', data_path)
        data = load_data(data_path, scale_params)
    
    model = load_model(model_ckpt)

    for l in model.layers: l.trainable = False

    # print(names_to_train)
    for l in model.layers:
        if l.name in names_to_train:
            l.trainable = True

    train_data = data['train']
    train_fluorescence = train_data['fluorescence']
    train_fluorescence = np.expand_dims(train_fluorescence, axis=-1)
    train_op = np.stack([train_data['mu_a'], train_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    train_depth = train_data['depth']
    train_depth[train_depth == 0] = 10
    train_concentration_fluor = train_data['concentration_fluor']
    train_reflectance = train_data['reflectance']

    train_dataset = tf.data.Dataset.from_tensor_slices((
        (train_op, train_fluorescence),  # tuple of inputs
        {'outDF': train_depth, 'outQF': train_concentration_fluor}  # dict of outputs
    ))
    train_dataset = train_dataset.shuffle(buffer_size=1000, seed=1024, reshuffle_each_iteration=False).batch(batch)

    val_data = data['val']
    val_fluorescence = val_data['fluorescence']
    val_fluorescence = np.expand_dims(val_fluorescence, axis=-1)
    val_op = np.stack([val_data['mu_a'], val_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    val_depth = val_data['depth']
    val_depth[val_depth == 0] = 10
    val_concentration_fluor = val_data['concentration_fluor']
    val_reflectance = val_data['reflectance']

    val_dataset = tf.data.Dataset.from_tensor_slices((
        (val_op, val_fluorescence),  # tuple of inputs
        {'outDF': val_depth, 'outQF': val_concentration_fluor}  # dict of outputs
    ))
    val_dataset = val_dataset.batch(batch)

    test_data = data['test']
    test_fluorescence = test_data['fluorescence']
    test_fluorescence = np.expand_dims(test_fluorescence, axis=-1)
    test_op = np.stack([test_data['mu_a'], test_data['mu_s']], axis=1).transpose(0, 2, 3, 1)
    test_depth = test_data['depth']
    test_depth[test_depth == 0] = 10
    test_concentration_fluor = test_data['concentration_fluor']
    test_reflectance = test_data['reflectance']

    test_dataset = tf.data.Dataset.from_tensor_slices((
        (test_op, test_fluorescence),  # tuple of inputs
        {'outDF': test_depth, 'outQF': test_concentration_fluor}  # dict of outputs
    ))
    test_dataset = test_dataset.batch(batch)

    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=decay_rate, patience=patience, verbose=1, min_lr=1e-6)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=patience, verbose=1, mode='auto')
    batch_logger = BatchLogger(log_interval=20, num_samples= int(train_fluorescence.shape[0]), batch_size=batch)

    if is_aws:
        model_dir = f'/opt/ml/model'
    else:
        model_dir = params['model_dir']
        os.makedirs(model_dir, exist_ok=True)

    checkpoint = CustomModelCheckpoint(
        filepath=model_dir,
        model_wrapper=model,
        monitor='val_loss',
        verbose=1
    )
    csv_logger = callbacks.CSVLogger(os.path.join(model_dir, f'loss.log'), append=True)
    cb = [lr_scheduler, early_stopping, batch_logger, checkpoint, csv_logger]

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss={'outQF': 'mae', 'outDF': 'mae'},
        optimizer=optimizer,
        metrics={
            'outQF': metrics.MeanAbsoluteError(name='mae_qf'),
            'outDF': metrics.MeanAbsoluteError(name='mae_df')
        }
    )

    print("++++++++ Keras version: ", keras.__version__, keras.__file__)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        verbose=0,
        callbacks=cb
    )

    eval_model = keras.models.load_model(os.path.join(model_dir, f'model.keras'))
    eval_model.evaluate(
        test_dataset,
        verbose=1
    )

if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()
    params = vars(args)
    
    names_to_train = [name.strip() for name in params['names_to_train'].split(',')]
    params['names_to_train'] = names_to_train

    np.random.seed(1024)
    tf.random.set_seed(1024)

    transfer_learning(params)