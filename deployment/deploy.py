from sagemaker.sklearn.model import SKLearnModel

def deploy_model(role, model_path):
    model = SKLearnModel(
        model_data=model_path,
        role=role,
        entry_point="src/inference/model_fn.py"
    )

    predictor = model.deploy(
        instance_type="ml.m5.large",
        initial_instance_count=1
    )

    return predictor


from sagemaker.sklearn.model import SKLearnModel
import sagemaker

role = sagemaker.get_execution_role()
session = sagemaker.Session()

model = SKLearnModel(
    model_data="s3://path-to-model/model.tar.gz",
    role=role,
    entry_point="src/inference/model_fn.py",
    framework_version="1.2-1",
    py_version="py3"
)

predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1
)

