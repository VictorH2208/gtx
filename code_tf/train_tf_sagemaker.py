from sagemaker.tensorflow import TensorFlow
from datetime import datetime

role = 'arn:aws:iam::425873948573:role/service-role/AmazonSageMaker-ExecutionRole-20220524T140113'

seed = 1024
train_subset = 1000
activation = "relu"
optimizer = "Adam"
epochs = 1
nF = 6
learningRate = 5e-4
batch = 32
xX = 101
yY = 101
decayRate = 0.4
depth_padding = 10
normalize = 1
fx_idx = "0 1 2 3 4 5"

scaleFL = 10e4
scaleOP0 = 10
scaleOP1 = 1
scaleDF = 1
scaleQF = 1
scaleRE = 1
nFilters3D = 128
kernelConv3D = "3 3 3"
strideConv3D = "1 1 1"
nFilters2D = 128
kernelConv2D = "3 3"
strideConv2D = "1 1"

data_path = "ts_2d_10000_original_TBR.mat" # <--- Change this if you want to use a different dataset
bucket_name = "20250509-victor" # <--- Change this to your own bucket

estimator = TensorFlow(
    entry_point='train_tf.py',     # Entry script
    source_dir='.',                   # This is the key: run from `code_tf/`
    role=role,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    framework_version='2.16',
    py_version='py310',
    script_mode=True,
    dependencies=['requirements.txt'],
    hyperparameters= {
        "seed": seed,
        "sagemaker": True,
        "train_subset": train_subset,
        "activation": "relu",
        "optimizer": "Adam",
        "epochs": epochs,
        "nF": nF,
        "learningRate": learningRate,
        "batch": batch,
        "xX": xX,
        "yY": yY,
        "decayRate": decayRate,
        "normalize": normalize,
        "depth_padding": depth_padding,
        "fx_idx": fx_idx,
        "scaleFL": scaleFL,
        "scaleOP0": scaleOP0,
        "scaleOP1": scaleOP1,
        "scaleDF": scaleDF,
        "scaleQF": scaleQF,
        "scaleRE": scaleRE,
        "nFilters3D": nFilters3D,
        "kernelConv3D": kernelConv3D,
        "strideConv3D": strideConv3D,
        "nFilters2D": nFilters2D,
        "kernelConv2D": kernelConv2D,
        "strideConv2D": strideConv2D,
        "data_path": data_path
    },
    output_path=f's3://{bucket_name}/tf_training_output/' # <--- Change this to your own bucket and output path
)
# job_name = f'vvv-tfTrain-{train_subset}-{seed}-{"_".join(data_path.split(".")[0].split("_")[:2])}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
# job_name = job_name.replace("_", "-")
job_name = "vvvv-test-model-hikaru-mod-morning3"
inputs = {
    'training': f's3://{bucket_name}/python_training_data_sim/{data_path}',
}
estimator.fit(inputs=inputs, job_name=job_name)
