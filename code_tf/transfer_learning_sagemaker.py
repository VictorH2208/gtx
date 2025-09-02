from sagemaker.tensorflow import TensorFlow
from datetime import datetime

role = 'arn:aws:iam::425873948573:role/service-role/AmazonSageMaker-ExecutionRole-20220524T140113'
names_list = ['conv2d_25', 'conv2d_26', 'conv2d_27', 'outQF_logits', 'conv2d_28', 'conv2d_29', 'outDF_logits']
names_string = ','.join(names_list)

estimator = TensorFlow(
    entry_point='transfer_learning.py',
    source_dir='.',
    role=role,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    framework_version='2.16',
    py_version='py310',
    script_mode=True,
    dependencies=['requirements.txt'],
    hyperparameters= {
        "is_aws": True,
        "model_path": "model.keras",
        "model_dir": "aws_ckpt",
        "data_path": "20250822_mcx_sujit_100scale_splited.mat",
        "names_to_train": names_string,
        "is_aws": True,
        "batch": 32,
        "epochs": 30,
        "learningRate": 1e-5,
        "decayRate": 0.4,
        "patience": 5,
        "scaleFL": 1e5,
        "scaleOP0": 10,
        "scaleOP1": 1,
        "scaleDF": 1,
        "scaleQF": 1,
        "scaleRE": 1,
    },
    output_path='s3://20250509-victor/transfer_output/'
)
job_name = f'tf-transfer-learning-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
inputs = {
    'training': 's3://20250509-victor/python_training_data_sim/20250822_mcx_sujit_100scale_splited.mat',
    'ckpt': 's3://20250509-victor/saved_model/unet_3d/model.keras',
}
estimator.fit(inputs=inputs, job_name=job_name)