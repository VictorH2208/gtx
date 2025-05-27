from sagemaker.tensorflow import TensorFlow
from datetime import datetime

role = 'arn:aws:iam::425873948573:role/service-role/AmazonSageMaker-ExecutionRole-20220524T140113'  # Replace this

estimator = TensorFlow(
    entry_point='train_tf.py',     # Entry script
    source_dir='.',                   # This is the key: run from `tensorflow/`
    role=role,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    framework_version='2.13',
    py_version='py310',
    script_mode=True,
    dependencies=['requirements.txt'],
    hyperparameters= {
        "sagemaker": True,
        "activation": "relu",
        "optimizer": "Adam",
        "epochs": 100,
        "nF": 6,
        "learningRate": 5e-4,
        "batch": 32,
        "xX": 101,
        "yY": 101,
        "decayRate": 0.4,
        "scaleFL": 1e5,
        "scaleOP0": 10,
        "scaleOP1": 1,
        "scaleDF": 1,
        "scaleQF": 1,
        "scaleRE": 1,
        "nFilters3D": 128,
        "kernelConv3D": "3 3 3",
        "strideConv3D": "1 1 1",
        "nFilters2D": 128,
        "kernelConv2D": "3 3",
        "strideConv2D": "1 1",
        "data_path": "s3://20250509-victor/python_training_data_sim/20241118_data_splited.mat"
    },
    output_path='s3://20250509-victor/tf_model_output/'
)
job_name = f'tf-model-output-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
inputs = {
    'training': 's3://20250509-victor/python_training_data_sim/20241118_data_splited.mat',
}
estimator.fit(inputs=inputs, job_name=job_name)
