#!/usr/bin/env python3
"""
Find index of target date in dates array
Converted from MATLAB libs/indexOfDate.m
"""

import pandas as pd

def index_of_date(dates, target_date):
    """
    Find the index in the dates array of the target date.
    
    Args:
        dates: pandas Series or array of dates
        target_date: target date string in 'YYYY-MM-DD' format
    
    Returns:
        int: index of the target date
    """
    target_dt = pd.to_datetime(target_date)
    matches = dates == target_dt
    
    if not matches.any():
        raise ValueError(f'Date {target_date} not found')
    
    return matches.idxmax()