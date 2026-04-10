"""
src — Data-Driven Dynamic Pricing System
=========================================
This package contains:
  - model_training  : ML pipeline (train, compare, save models)
  - pricing_engine  : Optimal price recommendation engine
"""

from src.model_training import run_pipeline, load_and_preprocess_data, engineer_features
from src.pricing_engine import PricingEngine

__version__ = "1.0.0"
__author__ = "Riya"

__all__ = [
    "run_pipeline",
    "load_and_preprocess_data",
    "engineer_features",
    "PricingEngine",
]
