"""
Unit tests for strategy
"""

import unittest
import pandas as pd
import numpy as np
from strategy import Strategy


class TestStrategy(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'parameters': {
                'lookback_period': 20,
                'entry_threshold': 2.0,
                'exit_threshold': 0.5
            },
            'risk': {
                'max_position_size': 0.1,
                'max_drawdown_exit': 0.15
            }
        }
        self.strategy = Strategy(self.config)

        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': np.random.randn(100).cumsum() + 100,
            'high': np.random.randn(100).cumsum() + 102,
            'low': np.random.randn(100).cumsum() + 98,
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertIsNotNone(self.strategy)
        self.assertEqual(self.strategy.position, 0)
        self.assertEqual(self.strategy.portfolio_value, 100000)

    def test_signal_generation(self):
        """Test signal generation."""
        signals = self.strategy.generate_signals(self.test_data)
        self.assertEqual(len(signals), len(self.test_data))
        self.assertTrue(all(s in [-1, 0, 1] for s in signals))

    def test_position_sizing(self):
        """Test position size calculation."""
        size = self.strategy.calculate_position_size(1, 100)
        self.assertGreater(size, 0)
        self.assertLessEqual(size * 100, self.strategy.portfolio_value * 0.1)

    def test_execute_trade(self):
        """Test trade execution."""
        initial_position = self.strategy.position
        self.strategy.execute_trade(1, 100, pd.Timestamp('2020-01-01'))
        self.assertNotEqual(self.strategy.position, initial_position)


if __name__ == '__main__':
    unittest.main()
