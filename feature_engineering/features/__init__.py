"""
NBA PRA Feature Engineering - Feature Calculation Modules

This package contains all feature category calculators.
Each module is independent and generates a specific set of features.

All modules maintain the data grain: [player_id, game_id, game_date]
All modules prevent temporal leakage using .shift(1) for rolling calculations.
"""

__all__ = [
    'rolling_features',
    'matchup_features',
    'contextual_features',
    'advanced_metrics',
    'position_features',
    'injury_features',
]

__version__ = "2.0.0"
