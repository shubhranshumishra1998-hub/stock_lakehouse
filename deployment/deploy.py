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
