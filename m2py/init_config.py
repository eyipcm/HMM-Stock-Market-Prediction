#!/usr/bin/env python3
"""
Configuration initialization for HMM Stock Market Prediction
Converted from MATLAB init.m
"""

import pandas as pd
from datetime import datetime

def init_config():
    """Initialize configuration parameters"""
    print('Init')
    
    config = {
        'stock_name': 'AAPL.csv',
        'TRAIN': 0,  # Use pre-trained model;   Training Mode (TRAIN = 1), Prediction Mode (TRAIN = 0)
        'filename': 'train/hmmtrain-20250904184643.pkl',
        
        # Training sequence parameters
        'shift_window_by_one': 1,  # 0: group by latency, 1: shift by 1 day
        
        # Training period
        'start_train_date': '2017-01-03',
        'end_train_date': '2019-01-03',
        
        # Prediction period
        'start_prediction_date': '2023-01-03',
        'end_prediction_date': None,  # Will be set to last available date
        
        # Dynamic edges for discretization
        'use_dynamic_edges': 0,  # 0: default values, 1: dynamic based on training set
        
        # Discretization parameters
        'discretization_points': [50, 10, 10],  # [fracChange, fracHigh, fracLow]
        'total_discretization_points': 2518,  # Match the trained model
        
        # HMM parameters
        'underlying_states': 4,  # number of hidden states
        'mixtures_number': 4,    # number of mixture components for each state
        'latency': 10           # days aka vectors in sequence
    }
    
    # Load stock data to get date indices
    stock_data = load_stock_data(config['stock_name'])
    
    # Set training date indices
    config['start_train_date_idx'] = index_of_date(stock_data['Date'], config['start_train_date'])
    config['end_train_date_idx'] = index_of_date(stock_data['Date'], config['end_train_date'])
    config['train_indexes'] = range(config['start_train_date_idx'], config['end_train_date_idx'] + 1)
    
    # Set prediction date indices
    config['start_prediction_date_idx'] = index_of_date(stock_data['Date'], config['start_prediction_date'])
    if config['end_prediction_date'] is None:
        config['end_prediction_date_idx'] = len(stock_data) - 1
    else:
        config['end_prediction_date_idx'] = index_of_date(stock_data['Date'], config['end_prediction_date'])
    
    config['prediction_length'] = config['end_prediction_date_idx'] - config['start_prediction_date_idx'] + 1
    config['prediction_indexes'] = range(config['start_prediction_date_idx'], config['end_prediction_date_idx'] + 1)
    
    return config

def load_stock_data(stock_name):
    """Load stock data from CSV file"""
    csv_path = f'../datasets/csv/{stock_name}'
    data = pd.read_csv(csv_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def index_of_date(dates, target_date):
    """Find index of target date in dates array"""
    target_dt = pd.to_datetime(target_date)
    matches = dates == target_dt
    if not matches.any():
        raise ValueError(f'Date {target_date} not found')
    return matches.idxmax()