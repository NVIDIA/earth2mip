from earth2mip._config import Settings
from earth2mip.model_registry import ModelRegistry

__version__ = "23.8.15"

config = Settings()
registry = ModelRegistry(config.MODEL_REGISTRY)
