import pandas as pd
import numpy as np

class Q1500MeanReversion:
    """
    Q1500 Mean Reversion Strategy
    
    Logic:
    1. Rank stocks by N-day returns.
    2. Long the bottom X% (losers).
    3. Short the top Y% (winners).
    4. Hold for 1 week (or rebalance weekly).
    """
    def __init__(self, lookback=5, long_pct=10, short_pct=90):
        self.lookback = lookback
        self.long_pct = long_pct
        self.short_pct = short_pct

    def generate_signals(self, prices):
        """
        Generate target weights based on historical prices.
        
        Args:
            prices (pd.DataFrame): DataFrame of asset prices (index=date, columns=symbols).
            
        Returns:
            pd.Series: Target weights for each asset.
        """
        # Calculate returns over the lookback period
        # We use the last 'lookback' days to determine the signal for TODAY
        if len(prices) < self.lookback + 1:
            return pd.Series(0, index=prices.columns)

        # Calculate N-day returns
        # (Price_t - Price_{t-N}) / Price_{t-N}
        returns = prices.pct_change(self.lookback)
        
        # We are interested in the most recent return signal
        last_returns = returns.iloc[-1].dropna()
        
        if last_returns.empty:
             return pd.Series(0, index=prices.columns)

        # Determine thresholds
        try:
            long_threshold = np.percentile(last_returns, self.long_pct)
            short_threshold = np.percentile(last_returns, self.short_pct)
        except IndexError:
            return pd.Series(0, index=prices.columns)
        
        # Identify assets
        longs = last_returns[last_returns <= long_threshold].index
        shorts = last_returns[last_returns >= short_threshold].index
        
        # Construct weights
        weights = pd.Series(0.0, index=prices.columns)
        
        # Equal weight long bucket (total 0.5 leverage)
        if len(longs) > 0:
            weights[longs] = 0.5 / len(longs)
        
        # Equal weight short bucket (total -0.5 leverage)
        if len(shorts) > 0:
            weights[shorts] = -0.5 / len(shorts)
            
        return weights