import sagemaker
from sagemaker.sklearn.estimator import SKLearn

role = sagemaker.get_execution_role()
session = sagemaker.Session()

estimator = SKLearn(
    entry_point="src/training/train.py",
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",
    py_version="py3",
    hyperparameters={}
)

estimator.fit({
    "train": "s3://bucket2-curated-stock/ml/train/"
})
