import pandas as pd

import numpy as np


def evaluate_forecast(y_true, y_pred):
    """
    评估气象预测结果的性能，计算 MAE、MSE 和 RMSE。
    Args:
        y_true : 真实的目标值。
        y_pred : 预测的目标值。
    Returns:
        MAE、MSE 和 RMSE 的评估结果。
    """
    # Convert to NumPy arrays in case inputs are pandas Series
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate MAE, MSE, and RMSE
    mae = np.mean(np.abs(y_true - y_pred), axis=0)
    mse = np.mean((y_true - y_pred) ** 2, axis=0)
    rmse = np.sqrt(mse)

    return mae, mse, rmse
