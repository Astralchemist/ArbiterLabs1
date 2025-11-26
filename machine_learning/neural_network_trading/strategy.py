import numpy as np
import pandas as pd

class NeuralNetworkStrategy:
    def __init__(self, params=None):
        self.params = params or {}
        self.means = np.zeros(5)
        self.stds = np.ones(5)
        
        self.weights_hidden = np.zeros((5, 8))
        self.weights_output = np.zeros((8, 1))
        
        if params:
            self._initialize_weights(params)
    
    def _initialize_weights(self, params):
        for i in range(5):
            for j in range(8):
                key = f'w_i{i+1}_h{j+1}'
                if key in params:
                    self.weights_hidden[i][j] = params[key]
        
        for i in range(8):
            key = f'w_h{i+1}_o1'
            if key in params:
                self.weights_output[i][0] = params[key]
    
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def forward(self, inputs):
        inputs = np.array(inputs)
        hidden_activations = self.sigmoid(np.dot(inputs, self.weights_hidden))
        output_activations = self.sigmoid(np.dot(hidden_activations, self.weights_output))
        return output_activations
    
    def scale_data(self, data):
        inputs = np.array(data)
        scaled_inputs = (inputs - self.means) / self.stds
        return scaled_inputs
    
    def update_statistics(self, data):
        self.means = np.array([np.mean([x, self.means[i]]) for i, x in enumerate(data)])
        self.stds = np.array([np.std([x, self.stds[i]]) for i, x in enumerate(data)])
    
    def predict(self, features):
        self.update_statistics(features)
        scaled_inputs = self.scale_data(features)
        output = self.forward(scaled_inputs)
        return 'buy' if output > 0.5 else 'sell'

def calculate_indicators(df):
    df['sma'] = df['close'].rolling(window=15).mean()
    df['atr'] = calculate_atr(df, 14)
    df['rsi'] = calculate_rsi(df['close'], 14)
    return df

def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
