from sagemaker.tensorflow import TensorFlow

role = 'arn:aws:iam::425873948573:role/service-role/AmazonSageMaker-ExecutionRole-20220524T140113'  # Replace this

seed = 1024
train_subset = 8000
activation = "relu"
optimizer = "Adam"
epochs = 100
nF = 6
learningRate = 5e-4
batch = 32
xX = 101
yY = 101
decayRate = 0.4
normalize = True
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
data_path = "ts_2d_10000.mat"

estimator = TensorFlow(
    entry_point='train_tf.py',     # Entry script
    source_dir='.',                   # This is the key: run from `tensorflow/`
    role=role,
    instance_count=1,
    instance_type='ml.g5.2xlarge',
    framework_version='2.16',
    py_version='py310',
    script_mode=True,
    dependencies=['requirements.txt'],
    hyperparameters= {
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
    output_path='s3://20250509-victor/tf_training_output/'
)
job_name = f'vvv-tfTrain-subset{train_subset}-epochs{epochs}-batch{batch}-data{data_path.split(".")[0]}'
job_name = job_name.replace("_", "-")
inputs = {
    'training': 's3://20250509-victor/python_training_data_sim/ts_2d_10000.mat',
}
estimator.fit(inputs=inputs, job_name=job_name)
