#!/usr/bin/env python3
"""
HMM observation prediction
Converted from MATLAB libs/hmmPredictObservation.m

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
from hmmlearn import hmm

def hmm_predict_observation(obs_seq, trans_matrix, emiss_matrix, 
                           verbose=0, possible_observations=None, dynamic_window=True):
    """
    Predicts the next observation in a Hidden Markov Model (HMM).
    
    Args:
        obs_seq: The sequence of observations for which the next observation is to be predicted
        trans_matrix: The transition matrix representing transition probabilities between states
        emiss_matrix: The emission matrix representing probabilities of emitting different observations
        verbose: Optional flag to enable verbose information printing. Default: 0
        possible_observations: A vector of possible observations for predicting the next observation
        dynamic_window: Optional flag to enable dynamic windowing for convergence. Default: True
    
    Returns:
        int: The predicted observation as the next one in the sequence
    """
    if possible_observations is None:
        # If no possible observations are specified, predict using the standard approach
        model = hmm.CategoricalHMM(n_components=trans_matrix.shape[0], init_params='')
        model.startprob_ = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]  # Add start probabilities
        model.transmat_ = trans_matrix
        model.emissionprob_ = emiss_matrix
        
        # Manually set the fitted flag to bypass the fit requirement
        model._is_fitted = True
        
        # Decode the sequence to get state probabilities
        log_prob, states = model.decode(obs_seq.reshape(-1, 1))
        # Get the state probabilities for the last observation
        last_state_probs = model.predict_proba(obs_seq.reshape(-1, 1))[-1]
        next_state_p = trans_matrix.T @ last_state_probs
        next_obs_p = emiss_matrix.T @ next_state_p
        
        predicted_observation = np.argmax(next_obs_p)
        
        if verbose:
            print(f"Log probability of sequence: {log_prob:.4f}")
            print(f"Observation probability: {next_obs_p[predicted_observation]:.4f}")
    else:
        # If a (sub)set of possible observations is available
        max_log_prob = -np.inf
        most_likely_obs = np.nan
        
        if dynamic_window:
            num_trials = len(obs_seq) - 3
        else:
            num_trials = 1
        
        current_seq = obs_seq.copy()
        
        for trials in range(num_trials):
            for possible_obs in possible_observations:
                # Create extended sequence
                extended_seq = np.append(current_seq, possible_obs).reshape(-1, 1)
                
                # Create model and decode
                model = hmm.CategoricalHMM(n_components=trans_matrix.shape[0], init_params='')
                model.startprob_ = np.ones(trans_matrix.shape[0]) / trans_matrix.shape[0]  # Add start probabilities
                model.transmat_ = trans_matrix
                model.emissionprob_ = emiss_matrix
                
                # Manually set the fitted flag to bypass the fit requirement
                model._is_fitted = True
                
                try:
                    log_prob, _ = model.decode(extended_seq)
                    
                    if max_log_prob < log_prob:
                        max_log_prob = log_prob
                        most_likely_obs = possible_obs
                except:
                    continue
            
            if max_log_prob == -np.inf:
                # If convergence is not reached, remove the first value from the sequence
                current_seq = current_seq[1:]
            else:
                break
        
        predicted_observation = most_likely_obs
        
        if verbose:
            print(f"Log probability of sequence: {max_log_prob:.4f}")
    
    return predicted_observation