# models/model_factory.py
from Anomaly_and_Shock_Detection_Weather.models.xgboost_model import XGBoostModel
from Anomaly_and_Shock_Detection_Weather.models.random_forest_model import RandomForestModel
from Anomaly_and_Shock_Detection_Weather.models.svm_model import SVMModel

class ModelFactory:
    MODEL_REGISTRY = {
        'xgboost': XGBoostModel,
        'random_forest': RandomForestModel,
        'svm': SVMModel,
    }

    @staticmethod
    def create_model(model_name, config, **kwargs):
        """Create a single model instance, passing model_name explicitly."""
        if model_name not in ModelFactory.MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'. Available: {list(ModelFactory.MODEL_REGISTRY.keys())}")
        model_class = ModelFactory.MODEL_REGISTRY[model_name]
        return model_class(config, model_name=model_name, **kwargs)

    @staticmethod
    def create_all_models(config):
        """Create instances of all registered models."""
        return {name: cls(config, model_name=name) for name, cls in ModelFactory.MODEL_REGISTRY.items()}

    @staticmethod
    def get_available_models():
        """Return list of available model names."""
        return list(ModelFactory.MODEL_REGISTRY.keys())