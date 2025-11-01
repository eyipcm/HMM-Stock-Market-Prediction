#!/usr/bin/env python3
"""
HMM Stock Market Prediction - Main Python Implementation
Converted from MATLAB main.m

SPDX-FileCopyrightText: Copyright (C) 2025 Ernest YIP <eyipcm@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import sys
from scipy import stats
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import pickle

# Add libs directory to path
sys.path.append('libs')

# Create output directories
os.makedirs('output_figs', exist_ok=True)
os.makedirs('train', exist_ok=True)

from init_config import init_config
from libs.dynamic_edges import dynamic_edges
from libs.index_of_date import index_of_date
from libs.map.map_3d_to_1d import map_3d_to_1d
from libs.map.map_1d_to_3d import map_1d_to_3d
from libs.hmm_predict_observation import hmm_predict_observation
from libs.rg_candle import rg_candle

def main():
    """Main function for HMM stock market prediction"""
    
    # Clear previous plots and warnings
    plt.close('all')
    warnings.filterwarnings('ignore')
    
    # Initialize configuration
    config = init_config()
    
    # Load stock data
    stock_data = load_stock_data(config['stock_name'])
    
    # Extract training data
    train_data = extract_training_data(stock_data, config)
    
    # Calculate fractional changes
    frac_change = (train_data['Close'] - train_data['Open']) / train_data['Open']
    frac_high = (train_data['High'] - train_data['Open']) / train_data['Open']
    frac_low = (train_data['Open'] - train_data['Low']) / train_data['Open']
    
    # Create 3D observations matrix
    continuous_observations_3d = np.column_stack([frac_change, frac_high, frac_low])
    
    # Set up discretization edges
    if config['use_dynamic_edges']:
        edges_f_change = dynamic_edges(frac_change, config['discretization_points'][0])
        edges_f_high = dynamic_edges(frac_high, config['discretization_points'][1])
        edges_f_low = dynamic_edges(frac_low, config['discretization_points'][2])
    else:
        edges_f_change = np.linspace(-0.1, 0.1, config['discretization_points'][0] + 1)
        edges_f_high = np.linspace(0, 0.1, config['discretization_points'][1] + 1)
        edges_f_low = np.linspace(0, 0.1, config['discretization_points'][2] + 1)
    
    # Discretize observations
    frac_change_discrete = np.digitize(frac_change, edges_f_change) - 1
    frac_high_discrete = np.digitize(frac_high, edges_f_high) - 1
    frac_low_discrete = np.digitize(frac_low, edges_f_low) - 1
    
    # Map to 1D discrete observations
    discrete_observations_1d = np.zeros(len(train_data))
    for i in range(len(train_data)):
        discrete_observations_1d[i] = map_3d_to_1d(
            frac_change_discrete[i] + 1,  # Convert to 1-based indexing
            frac_high_discrete[i] + 1,
            frac_low_discrete[i] + 1,
            config['discretization_points'][0],
            config['discretization_points'][1],
            config['discretization_points'][2]
        )
    
    # Training phase
    if config['TRAIN']:
        print('Markov Chain guesses')
        
        # Initialize transition matrix
        transition_matrix = np.ones((config['underlying_states'], config['underlying_states'])) / config['underlying_states']
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=config['mixtures_number'] * config['underlying_states'],
            covariance_type='diag',
            reg_covar=1e-10,
            random_state=42
        )
        gmm.fit(continuous_observations_3d)
        
        # Sort Gaussian components
        sorted_indices = np.argsort(gmm.means_[:, 0])
        sorted_means = gmm.means_[sorted_indices]
        sorted_covariances = gmm.covariances_[sorted_indices]
        
        # Initialize emission probabilities
        emission_probabilities = np.zeros((config['underlying_states'], config['total_discretization_points']))
        
        # Assign Gaussian Mixture for each hidden state
        for i in range(config['underlying_states']):
            start_idx = i * config['mixtures_number']
            end_idx = (i + 1) * config['mixtures_number']
            
            state_means = sorted_means[start_idx:end_idx]
            state_covs = sorted_covariances[start_idx:end_idx]
            
            # Calculate emission probabilities for each discrete observation
            for x_idx, x_val in enumerate(edges_f_change[:-1]):
                for y_idx, y_val in enumerate(edges_f_high[:-1]):
                    for z_idx, z_val in enumerate(edges_f_low[:-1]):
                        # Calculate probability density
                        prob = 0
                        for k in range(len(state_means)):
                            prob += gmm.weights_[sorted_indices[start_idx + k]] * \
                                   multivariate_normal_pdf([x_val, y_val, z_val], 
                                                          state_means[k], 
                                                          state_covs[k])
                        
                        # Map to 1D index
                        emission_idx = map_3d_to_1d(x_idx + 1, y_idx + 1, z_idx + 1,
                                                  config['discretization_points'][0],
                                                  config['discretization_points'][1],
                                                  config['discretization_points'][2])
                        emission_probabilities[i, int(emission_idx) - 1] = prob
            
            # Normalize probabilities
            emission_probabilities[i, :] = emission_probabilities[i, :] / np.sum(emission_probabilities[i, :])
        
        # Create training sequences
        if config['shift_window_by_one']:
            total_train_sequences = len(train_data) - config['latency'] + 1
            training_set = np.zeros((total_train_sequences, config['latency']))
            for i in range(total_train_sequences):
                training_set[i, :] = discrete_observations_1d[i:i + config['latency']]
        else:
            total_train_sequences = len(train_data) // config['latency']
            training_set = np.zeros((total_train_sequences, config['latency']))
            for i in range(total_train_sequences):
                start_idx = i * config['latency']
                end_idx = start_idx + config['latency']
                training_set[i, :] = discrete_observations_1d[start_idx:end_idx]
        
        # Train HMM
        print('Training HMM...')
        model = hmm.CategoricalHMM(n_components=config['underlying_states'], n_iter=1000)
        model.startprob_ = np.ones(config['underlying_states']) / config['underlying_states']
        model.transmat_ = transition_matrix
        model.emissionprob_ = emission_probabilities
        
        # Convert training data to proper format for hmmlearn
        training_data = training_set.astype(int) - 1  # Convert to 0-based indexing
        
        # Train the model
        start_time = datetime.now()
        model.fit(training_data)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Save trained model
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f'train/hmmtrain-{timestamp}.pkl'
        os.makedirs('train', exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': model,
                'edges_f_change': edges_f_change,
                'edges_f_high': edges_f_high,
                'edges_f_low': edges_f_low,
                'training_time': training_time,
                'config': config
            }, f)
        
        print(f'Model saved to {filename}')
        print(f'Training time: {training_time:.2f} seconds')
    
    else:
        # Load pre-trained model
        with open(config['filename'], 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            edges_f_change = model_data['edges_f_change']
            edges_f_high = model_data['edges_f_high']
            edges_f_low = model_data['edges_f_low']
    
    # Prediction phase
    print('Making predictions...')
    predicted_observations_3d = np.zeros((config['prediction_length'], 3))
    predicted_close = np.zeros(config['prediction_length'])
    
    for current_prediction in range(config['prediction_length']):
        # Historical window indexes
        current_window_start_idx = config['start_prediction_date_idx'] - config['latency'] + current_prediction
        current_window_end_idx = config['start_prediction_date_idx'] - 2 + current_prediction
        
        current_prediction_idx = current_window_end_idx + 1
        current_window_indexes = range(current_window_start_idx, current_window_end_idx + 1)
        
        # Historical window observations
        window_data = stock_data.iloc[current_window_indexes]
        current_window_frac_change = (window_data['Close'] - window_data['Open']) / window_data['Open']
        current_window_frac_high = (window_data['High'] - window_data['Open']) / window_data['Open']
        current_window_frac_low = (window_data['Open'] - window_data['Low']) / window_data['Open']
        
        # Discretize historical data
        current_window_frac_change_discrete = np.digitize(current_window_frac_change, edges_f_change) - 1
        current_window_frac_high_discrete = np.digitize(current_window_frac_high, edges_f_high) - 1
        current_window_frac_low_discrete = np.digitize(current_window_frac_low, edges_f_low) - 1
        
        # Map to 1D
        current_window_1d = np.zeros(config['latency'] - 1)
        for i in range(config['latency'] - 1):
            current_window_1d[i] = map_3d_to_1d(
                current_window_frac_change_discrete[i] + 1,
                current_window_frac_high_discrete[i] + 1,
                current_window_frac_low_discrete[i] + 1,
                config['discretization_points'][0],
                config['discretization_points'][1],
                config['discretization_points'][2]
            )
        
        # Make prediction
        progress = (current_prediction + 1) / config['prediction_length'] * 100
        print(f'{progress:.2f}% : ', end='')
        
        predicted_observation_1d = hmm_predict_observation(
            current_window_1d.astype(int) - 1,  # Convert to 0-based indexing
            model.transmat_,
            model.emissionprob_,
            possible_observations=range(config['total_discretization_points']),
            verbose=1
        )
        
        if not np.isnan(predicted_observation_1d):
            # Map back to 3D
            predicted_frac_change_idx, predicted_frac_high_idx, predicted_frac_low_idx = map_1d_to_3d(
                int(predicted_observation_1d) + 1,  # Convert to 1-based indexing
                config['discretization_points'][0],
                config['discretization_points'][1],
                config['discretization_points'][2]
            )
            
            predicted_observations_3d[current_prediction, :] = [
                edges_f_change[predicted_frac_change_idx - 1],
                edges_f_high[predicted_frac_high_idx - 1],
                edges_f_low[predicted_frac_low_idx - 1]
            ]
            
            predicted_close[current_prediction] = stock_data.iloc[current_prediction_idx]['Open'] * \
                                                (1 + predicted_observations_3d[current_prediction, 0])
        else:
            predicted_observations_3d[current_prediction, :] = np.nan
            predicted_close[current_prediction] = np.nan
    
    # Generate plots and analysis
    generate_plots_and_analysis(stock_data, predicted_close, config)
    
    print('Analysis complete!')

def load_stock_data(stock_name):
    """Load stock data from CSV file"""
    csv_path = f'../datasets/csv/{stock_name}'
    data = pd.read_csv(csv_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def extract_training_data(stock_data, config):
    """Extract training data based on configuration"""
    train_start_idx = config['start_train_date_idx']
    train_end_idx = config['end_train_date_idx']
    return stock_data.iloc[train_start_idx:train_end_idx + 1]

def multivariate_normal_pdf(x, mean, cov):
    """Calculate multivariate normal probability density function"""
    try:
        return stats.multivariate_normal.pdf(x, mean, cov)
    except:
        return 0.0

def generate_plots_and_analysis(stock_data, predicted_close, config):
    """Generate plots and perform analysis"""
    prediction_data = stock_data.iloc[config['prediction_indexes']]
    
    # Calculate MAPE and DPA
    valid_predictions = ~np.isnan(predicted_close)
    if np.any(valid_predictions):
        actual_close = prediction_data['Close'].values[valid_predictions]
        predicted_close_valid = predicted_close[valid_predictions]
        
        # Calculate MAPE
        mape = np.mean(np.abs((actual_close - predicted_close_valid) / actual_close)) * 100
        
        # Calculate DPA (Directional Prediction Accuracy)
        actual_direction = np.sign(prediction_data['Close'].values[valid_predictions] - 
                                 prediction_data['Open'].values[valid_predictions])
        predicted_direction = np.sign(predicted_close_valid - prediction_data['Open'].values[valid_predictions])
        dpa = np.mean(actual_direction == predicted_direction) * 100
        
        print(f'\nMean Absolute Percentage Error (MAPE): {mape:.2f}%')
        print(f'Directional Prediction Accuracy (DPA): {dpa:.2f}%')
    
    # Create plots
    create_candlestick_plot(prediction_data, predicted_close, config)
    create_prediction_plot(prediction_data, predicted_close, config)

def create_candlestick_plot(prediction_data, predicted_close, config):
    """Create candlestick plot with predictions"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot candlestick chart
    rg_candle(prediction_data, ax=ax)
    
    # Plot predictions using the same numeric positioning as candlesticks
    valid_predictions = ~np.isnan(predicted_close)
    if np.any(valid_predictions):
        actual_direction = np.sign(prediction_data['Close'].values[valid_predictions] - 
                                 prediction_data['Open'].values[valid_predictions])
        predicted_direction = np.sign(predicted_close[valid_predictions] - 
                                    prediction_data['Open'].values[valid_predictions])
        
        correct_direction = actual_direction == predicted_direction
        
        # Get the numeric positions for valid predictions (same as candlesticks)
        valid_indices = np.where(valid_predictions)[0]
        
        # Plot prediction arrows (showing direction)
        for i, pred_idx in enumerate(valid_indices):
            current_index = pred_idx
            current_prediction = predicted_close[pred_idx]
            current_close = prediction_data['Close'].iloc[pred_idx]
            current_open = prediction_data['Open'].iloc[pred_idx]
            
            # Determine if prediction direction was correct
            actual_direction = np.sign(current_close - current_open)
            predicted_direction = np.sign(current_prediction - current_open)
            is_correct = actual_direction == predicted_direction
            
            # Create arrow - connect from previous close to current prediction
            if i == 0:
                # For first prediction, just plot the point with arrow
                ax.annotate('', xy=(current_index, current_prediction), 
                           xytext=(current_index, current_prediction),
                           arrowprops=dict(arrowstyle='->', color='g' if is_correct else 'r', 
                                         lw=1.5, alpha=0.8))
            else:
                # Connect from previous actual close to current prediction with arrow
                prev_index = valid_indices[i-1]
                prev_close = prediction_data['Close'].iloc[prev_index]
                
                # Create arrow
                color = 'g' if is_correct else 'r'
                ax.annotate('', xy=(current_index, current_prediction), 
                           xytext=(prev_index, prev_close),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                         lw=1.5, alpha=0.8, shrinkA=0, shrinkB=0))
        
        # Create legend manually (like MATLAB does)
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch
        legend_elements = [
            Line2D([0], [0], color='g', marker='o', markersize=8, label='Close predictions (correct direction)'),
            Line2D([0], [0], color='r', marker='o', markersize=8, label='Close predictions (wrong direction)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(f"{config['stock_name'].replace('.csv', '')} - Candlestick Chart")
    ax.grid(True)
    plt.tight_layout()
    
    # Save plot to file
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(f'output_figs/{config["stock_name"].replace(".csv", "")}_candlestick_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_prediction_plot(prediction_data, predicted_close, config):
    """Create prediction comparison plot - matches MATLAB behavior"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual close prices
    ax.plot(prediction_data['Date'], prediction_data['Close'], 'b-', 
            linewidth=0.5, marker='.', markersize=3, label='Actual stock close')
    
    # Plot prediction arrows (showing direction)
    valid_predictions = ~np.isnan(predicted_close)
    if np.any(valid_predictions):
        # Get indices where we have valid predictions
        valid_indices = np.where(valid_predictions)[0]
        
        # For each valid prediction, create an arrow
        for i, pred_idx in enumerate(valid_indices):
            current_date = prediction_data['Date'].iloc[pred_idx]
            current_prediction = predicted_close[pred_idx]
            current_open = prediction_data['Open'].iloc[pred_idx]
            current_close = prediction_data['Close'].iloc[pred_idx]
            
            # Determine if prediction direction was correct
            actual_direction = np.sign(current_close - current_open)
            predicted_direction = np.sign(current_prediction - current_open)
            is_correct = actual_direction == predicted_direction
            
            # Create arrow - connect from previous close to current prediction
            if i == 0:
                # For first prediction, just plot the point with arrow
                ax.annotate('', xy=(current_date, current_prediction), 
                           xytext=(current_date, current_prediction),
                           arrowprops=dict(arrowstyle='->', color='g' if is_correct else 'r', 
                                         lw=1.5, alpha=0.8))
            else:
                # Connect from previous actual close to current prediction with arrow
                prev_date = prediction_data['Date'].iloc[valid_indices[i-1]]
                prev_close = prediction_data['Close'].iloc[valid_indices[i-1]]
                
                # Create arrow
                color = 'g' if is_correct else 'r'
                ax.annotate('', xy=(current_date, current_prediction), 
                           xytext=(prev_date, prev_close),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                         lw=1.5, alpha=0.8, shrinkA=0, shrinkB=0))
    
    # Create legend manually (like MATLAB does)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='b', marker='.', markersize=20, label='Actual stock close'),
        Line2D([0], [0], color='g', marker='o', markersize=8, label='Close predictions (correct direction)'),
        Line2D([0], [0], color='r', marker='o', markersize=8, label='Close predictions (wrong direction)')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    ax.set_title(f"{config['stock_name'].replace('.csv', '')} - Real vs predicted Close values")
    ax.grid(True)
    
    # Set y-axis limits like MATLAB
    y_min = prediction_data['Close'].min() * 0.95
    y_max = prediction_data['Close'].max() * 1.05
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    # Save plot to file
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    plt.savefig(f'output_figs/{config["stock_name"].replace(".csv", "")}_predictions_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()