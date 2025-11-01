"""
Data comparison utilities for HMM Stock Market Prediction
Compares Python-generated data with MATLAB-generated data

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

import os
import numpy as np
import pandas as pd
import json
from datetime import datetime

class DataComparison:
    def __init__(self, python_output_dir, matlab_output_dir, results_dir):
        self.python_output_dir = python_output_dir
        self.matlab_output_dir = matlab_output_dir
        self.results_dir = results_dir
        self.comparison_results = {}
        
    def compare_all_data(self, verbose=False):
        """Compare all available data between Python and MATLAB outputs"""
        print("Comparing data...")
        
        results = {}
        
        # Compare model files
        model_results = self._compare_model_files(verbose)
        results['models'] = model_results
        
        # Compare prediction data
        prediction_results = self._compare_prediction_data(verbose)
        results['predictions'] = prediction_results
        
        # Compare metrics
        metrics_results = self._compare_metrics(verbose)
        results['metrics'] = metrics_results
        
        # Save results
        self.comparison_results = results
        self._save_data_results(results)
        
        return results
    
    def _compare_model_files(self, verbose=False):
        """Compare HMM model files"""
        python_models = self._find_model_files(self.python_output_dir)
        matlab_models = self._find_model_files(self.matlab_output_dir)
        
        if verbose:
            print(f"Python models: {python_models}")
            print(f"MATLAB models: {matlab_models}")
        
        if not python_models or not matlab_models:
            return {'status': 'no_models', 'message': 'No model files found for comparison'}
        
        # For now, just report what we found
        return {
            'python_models': python_models,
            'matlab_models': matlab_models,
            'status': 'found',
            'timestamp': datetime.now().isoformat()
        }
    
    def _find_model_files(self, directory):
        """Find model files in a directory"""
        model_files = []
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(('.pkl', '.mat', '.joblib')):
                    model_files.append(os.path.join(directory, file))
        return model_files
    
    def _compare_prediction_data(self, verbose=False):
        """Compare prediction data"""
        # This would compare actual prediction results
        # For now, return a placeholder
        return {
            'status': 'placeholder',
            'message': 'Prediction data comparison not yet implemented',
            'timestamp': datetime.now().isoformat()
        }
    
    def _compare_metrics(self, verbose=False):
        """Compare performance metrics"""
        # This would compare MAPE, DPA, etc.
        # For now, return a placeholder
        return {
            'status': 'placeholder',
            'message': 'Metrics comparison not yet implemented',
            'timestamp': datetime.now().isoformat()
        }
    
    def _save_data_results(self, results):
        """Save data comparison results to JSON"""
        output_file = os.path.join(self.results_dir, 'data_comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Data comparison results saved to: {output_file}")