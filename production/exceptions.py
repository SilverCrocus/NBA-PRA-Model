"""
Custom Exceptions for Production System

Provides specific exception types for better error handling and debugging.

Author: NBA PRA Prediction System
Date: 2025-11-01
"""


class ProductionError(Exception):
    """Base exception for production system errors"""
    pass


class ModelNotFoundError(ProductionError):
    """Raised when no trained models are found"""
    pass


class FeatureDataError(ProductionError):
    """Raised when feature data is missing or invalid"""
    pass


class OddsAPIError(ProductionError):
    """Raised when odds fetching fails"""
    pass


class PredictionError(ProductionError):
    """Raised when prediction generation fails"""
    pass


class BettingEngineError(ProductionError):
    """Raised when betting decision calculation fails"""
    pass


class InsufficientDataError(ProductionError):
    """Raised when player has insufficient historical data"""
    pass
