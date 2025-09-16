from sagemaker.tensorflow import TensorFlow
from datetime import datetime

role = 'arn:aws:iam::425873948573:role/service-role/AmazonSageMaker-ExecutionRole-20220524T140113'
names_list = ['conv2d_25', 'conv2d_26', 'conv2d_27', 'outQF_logits', 'conv2d_28', 'conv2d_29', 'outDF_logits']
names_string = ','.join(names_list)

bucket_name = "20250509-victor" # <--- Change this to your own bucket
data_path = "20250822_mcx_sujit_100scale_splited.mat" # <--- Change this to your own dataset
is_aws = True
model_folder = "unet_3d" # <--- Change this to your own model folder
model_name = "model.keras" # <--- Change this to your own model name
local_model_dir = "aws_ckpt" # <--- Change this to your own local model directory
seed = 1024
normalize = False
train_subset = 1000
batch = 32
epochs = 50
learning_rate = 1e-4
decay_rate = 0.4
patience = 5
scale_fl = 1e5
scale_op0 = 10
scale_op1 = 1
scale_df = 1
scale_qf = 1
scale_re = 1


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
        "is_aws": is_aws,
        "model_name": model_name,
        "local_model_dir": local_model_dir,
        "data_path": data_path,
        "names_to_train": names_string,
        "seed": seed,
        "normalize": normalize,
        "train_subset": train_subset,
        "batch": batch,
        "epochs": epochs,
        "learningRate": learning_rate,
        "decayRate": decay_rate,
        "patience": patience,
        "scaleFL": scale_fl,
        "scaleOP0": scale_op0,
        "scaleOP1": scale_op1,
        "scaleDF": scale_df,
        "scaleQF": scale_qf,
        "scaleRE": scale_re,
    },
    output_path=f's3://{bucket_name}/transfer_output/'
)
job_name = f'tf-transfer-learning-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
inputs = {
    'training': f's3://{bucket_name}/python_training_data_sim/{data_path}.mat',
    'ckpt': f's3://{bucket_name}/saved_model/{model_folder}/{model_name}',
}
estimator.fit(inputs=inputs, job_name=job_name)