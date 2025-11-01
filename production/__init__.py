"""
NBA PRA Production Prediction System

Production deployment for daily NBA player PRA (Points + Rebounds + Assists) predictions
with Monte Carlo probabilistic forecasting and Kelly criterion bet sizing.

Author: NBA PRA Prediction System
Date: 2025-10-31
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "NBA PRA Prediction System"

from production.config import *
from production.model_trainer import ProductionModelTrainer, train_production_models
from production.odds_fetcher import OddsFetcher, fetch_tomorrow_odds
from production.predictor import ProductionPredictor, predict_tomorrow
from production.betting_engine import BettingEngine, generate_bets_from_predictions

__all__ = [
    'ProductionModelTrainer',
    'train_production_models',
    'OddsFetcher',
    'fetch_tomorrow_odds',
    'ProductionPredictor',
    'predict_tomorrow',
    'BettingEngine',
    'generate_bets_from_predictions'
]
