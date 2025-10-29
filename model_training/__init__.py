"""
Model Training Package
NBA PRA Prediction Model Training and Validation
"""

from model_training.models import XGBoostModel, LightGBMModel, BaseModel
from model_training.training import train_model, train_all_models
from model_training.utils import (
    setup_logger,
    load_split_data,
    prepare_features_target,
    calculate_regression_metrics,
)

__version__ = "1.0.0"

__all__ = [
    # Models
    "XGBoostModel",
    "LightGBMModel",
    "BaseModel",
    # Training
    "train_model",
    "train_all_models",
    # Utilities
    "setup_logger",
    "load_split_data",
    "prepare_features_target",
    "calculate_regression_metrics",
]
