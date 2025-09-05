#!/usr/bin/env python3
"""
Candlestick chart plotting
Converted from MATLAB libs/RG_candle.m
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def rg_candle(data, ax=None, color_up='g', color_down='r'):
    """
    Plot candlestick chart from OHLC data.
    
    Args:
        data: DataFrame with columns ['Date', 'Open', 'High', 'Low', 'Close']
        ax: matplotlib axis object (optional)
        color_up: Color for bullish candles (close > open)
        color_down: Color for bearish candles (close < open)
    
    Returns:
        matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Ensure data is sorted by date
    data = data.sort_values('Date').reset_index(drop=True)
    
    # Convert dates to numeric for plotting
    dates = pd.to_datetime(data['Date'])
    x_pos = np.arange(len(dates))
    
    # Plot high-low lines
    for i, (idx, row) in enumerate(data.iterrows()):
        ax.plot([x_pos[i], x_pos[i]], [row['Low'], row['High']], 'k-', linewidth=0.8)
    
    # Plot candlestick bodies
    for i, (idx, row) in enumerate(data.iterrows()):
        open_price = row['Open']
        close_price = row['Close']
        high_price = row['High']
        low_price = row['Low']
        
        # Determine color based on open vs close
        if close_price >= open_price:
            color = color_up
            body_bottom = open_price
            body_top = close_price
        else:
            color = color_down
            body_bottom = close_price
            body_top = open_price
        
        # Create rectangle for candlestick body
        body_height = body_top - body_bottom
        if body_height > 0:
            rect = patches.Rectangle((x_pos[i] - 0.4, body_bottom), 0.8, body_height, 
                                   facecolor=color, edgecolor=color, alpha=0.8)
            ax.add_patch(rect)
        else:
            # Doji - just a line
            ax.plot([x_pos[i] - 0.4, x_pos[i] + 0.4], [open_price, open_price], color=color, linewidth=2)
    
    # Set x-axis labels
    tick_indices = range(0, len(dates), max(1, len(dates) // 10))
    ax.set_xticks([x_pos[i] for i in tick_indices])
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in tick_indices], 
                      rotation=45)
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, alpha=0.3)
    
    return ax