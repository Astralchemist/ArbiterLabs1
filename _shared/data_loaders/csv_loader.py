"""
CSV Data Loader

Load historical data from CSV files with validation and preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union


def load_data(
    file_path: str,
    date_column: str = 'date',
    parse_dates: bool = True,
    index_col: Optional[str] = None,
    standardize_columns: bool = True
) -> pd.DataFrame:
    """
    Load data from CSV file.

    Args:
        file_path: Path to CSV file
        date_column: Name of date column
        parse_dates: Whether to parse dates
        index_col: Column to use as index
        standardize_columns: Convert column names to lowercase

    Returns:
        DataFrame with loaded data

    Example:
        >>> data = load_data('prices.csv', date_column='Date')
        >>> data = load_data('ohlcv.csv', index_col='timestamp')
    """
    if index_col is None:
        index_col = date_column

    data = pd.read_csv(
        file_path,
        parse_dates=[date_column] if parse_dates else False,
        index_col=index_col
    )

    if standardize_columns:
        data.columns = data.columns.str.lower()

    return data


def load_multiple_csv(
    file_paths: List[str],
    date_column: str = 'date'
) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSV files.

    Args:
        file_paths: List of CSV file paths
        date_column: Name of date column

    Returns:
        Dictionary mapping filename to DataFrame
    """
    data_dict = {}
    for path in file_paths:
        filename = Path(path).stem
        data_dict[filename] = load_data(path, date_column)

    return data_dict


def load_from_folder(
    folder_path: str,
    pattern: str = "*.csv",
    date_column: str = 'date'
) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a folder.

    Args:
        folder_path: Path to folder
        pattern: File pattern to match
        date_column: Name of date column

    Returns:
        Dictionary mapping filename to DataFrame
    """
    folder = Path(folder_path)
    csv_files = list(folder.glob(pattern))

    data_dict = {}
    for file_path in csv_files:
        symbol = file_path.stem
        data_dict[symbol] = load_data(str(file_path), date_column)

    return data_dict


def validate_ohlcv_columns(data: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required OHLCV columns.

    Args:
        data: DataFrame to validate

    Returns:
        True if valid, False otherwise
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    return all(col in data.columns for col in required_cols)


def validate_ohlcv_data(data: pd.DataFrame) -> Dict[str, Union[bool, List]]:
    """
    Comprehensive validation of OHLCV data.

    Args:
        data: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    issues = []

    if not validate_ohlcv_columns(data):
        return {
            'valid': False,
            'issues': ['Missing required OHLCV columns']
        }

    if (data['high'] < data['low']).any():
        issues.append('High < Low detected')

    if (data['close'] > data['high']).any():
        issues.append('Close > High detected')

    if (data['close'] < data['low']).any():
        issues.append('Close < Low detected')

    if (data['open'] > data['high']).any():
        issues.append('Open > High detected')

    if (data['open'] < data['low']).any():
        issues.append('Open < Low detected')

    if (data[['open', 'high', 'low', 'close']] <= 0).any().any():
        issues.append('Negative or zero prices detected')

    if (data['volume'] < 0).any():
        issues.append('Negative volume detected')

    missing = data.isnull().sum()
    if missing.any():
        issues.append(f'Missing values: {missing[missing > 0].to_dict()}')

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }


def clean_ohlcv_data(
    data: pd.DataFrame,
    drop_invalid: bool = True,
    fill_missing: bool = False
) -> pd.DataFrame:
    """
    Clean and fix common issues in OHLCV data.

    Args:
        data: DataFrame to clean
        drop_invalid: Drop rows with invalid data
        fill_missing: Forward fill missing values

    Returns:
        Cleaned DataFrame
    """
    df = data.copy()

    if fill_missing:
        df = df.fillna(method='ffill').fillna(method='bfill')

    if drop_invalid:
        df = df[df['high'] >= df['low']]
        df = df[df['close'] <= df['high']]
        df = df[df['close'] >= df['low']]
        df = df[df['open'] <= df['high']]
        df = df[df['open'] >= df['low']]
        df = df[df[['open', 'high', 'low', 'close']] > 0].dropna()
        df = df[df['volume'] >= 0]

    return df


def save_to_csv(
    data: pd.DataFrame,
    file_path: str,
    include_index: bool = True
) -> None:
    """
    Save DataFrame to CSV file.

    Args:
        data: DataFrame to save
        file_path: Output file path
        include_index: Include index in CSV
    """
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=include_index)


def convert_column_names(
    data: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Convert column names to standard format.

    Args:
        data: DataFrame with custom column names
        mapping: Custom mapping dict, if None uses common mappings

    Returns:
        DataFrame with standardized column names

    Example:
        >>> mapping = {'Open': 'open', 'Close': 'close'}
        >>> data = convert_column_names(data, mapping)
    """
    if mapping is None:
        mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Date': 'date',
            'Datetime': 'datetime',
            'Time': 'time',
            'Timestamp': 'timestamp',
        }

    df = data.copy()
    df = df.rename(columns=mapping)

    return df


def resample_data(
    data: pd.DataFrame,
    freq: str = 'D'
) -> pd.DataFrame:
    """
    Resample OHLCV data to different frequency.

    Args:
        data: OHLCV DataFrame with datetime index
        freq: Pandas frequency string (D, W, M, H, etc.)

    Returns:
        Resampled DataFrame
    """
    resampled = pd.DataFrame()
    resampled['open'] = data['open'].resample(freq).first()
    resampled['high'] = data['high'].resample(freq).max()
    resampled['low'] = data['low'].resample(freq).min()
    resampled['close'] = data['close'].resample(freq).last()
    resampled['volume'] = data['volume'].resample(freq).sum()

    return resampled.dropna()


def merge_multiple_sources(
    data_dict: Dict[str, pd.DataFrame],
    column: str = 'close',
    join: str = 'outer'
) -> pd.DataFrame:
    """
    Merge data from multiple sources/symbols.

    Args:
        data_dict: Dictionary of DataFrames
        column: Column to extract from each DataFrame
        join: Join type ('inner', 'outer', 'left', 'right')

    Returns:
        Merged DataFrame with symbols as columns
    """
    dfs = []
    for symbol, df in data_dict.items():
        if column in df.columns:
            temp_df = df[[column]].copy()
            temp_df.columns = [symbol]
            dfs.append(temp_df)

    if len(dfs) == 0:
        return pd.DataFrame()

    result = pd.concat(dfs, axis=1, join=join)
    return result
