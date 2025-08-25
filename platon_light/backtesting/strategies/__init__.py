"""
Strategies package for Platon Light backtesting framework.

This package contains various trading strategies that can be used with the backtesting engine.
"""

from .moving_average_crossover import MovingAverageCrossover

__all__ = [
    "MovingAverageCrossover",
]
