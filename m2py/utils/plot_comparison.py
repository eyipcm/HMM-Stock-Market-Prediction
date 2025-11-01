"""
Plot comparison utilities for HMM Stock Market Prediction
Compares Python-generated plots with MATLAB-generated plots

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
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import pandas as pd
from datetime import datetime
import json

class PlotComparison:
    def __init__(self, python_output_dir, matlab_output_dir, results_dir):
        self.python_output_dir = python_output_dir
        self.matlab_output_dir = matlab_output_dir
        self.results_dir = results_dir
        self.comparison_results = {}
        
    def compare_all_plots(self, verbose=False):
        """Compare all available plots between Python and MATLAB outputs"""
        print("Comparing plots...")
        
        # Find all plot files
        python_plots = self._find_plot_files(self.python_output_dir)
        matlab_plots = self._find_plot_files(self.matlab_output_dir)
        
        if verbose:
            print(f"Found {len(python_plots)} Python plots")
            print(f"Found {len(matlab_plots)} MATLAB plots")
        
        # Compare each plot type
        results = {}
        
        # Compare candlestick plots
        candlestick_results = self._compare_candlestick_plots(python_plots, matlab_plots, verbose)
        results['candlestick'] = candlestick_results
        
        # Compare prediction plots
        prediction_results = self._compare_prediction_plots(python_plots, matlab_plots, verbose)
        results['prediction'] = prediction_results
        
        # Save results
        self.comparison_results = results
        self._save_plot_results(results)
        
        return results
    
    def _find_plot_files(self, directory):
        """Find all plot files in a directory"""
        plot_files = []
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                    plot_files.append(os.path.join(directory, file))
        return plot_files
    
    def _compare_candlestick_plots(self, python_plots, matlab_plots, verbose=False):
        """Compare candlestick plots"""
        python_candle = [p for p in python_plots if 'candlestick' in p.lower()]
        matlab_candle = [m for m in matlab_plots if 'candle' in m.lower()]
        
        if verbose:
            print(f"Python candlestick plots: {python_candle}")
            print(f"MATLAB candlestick plots: {matlab_candle}")
        
        if not python_candle or not matlab_candle:
            return {'status': 'no_plots', 'message': 'No candlestick plots found for comparison'}
        
        # Compare the first available plots
        python_plot = python_candle[0]
        matlab_plot = matlab_candle[0]
        
        return self._compare_plot_files(python_plot, matlab_plot, 'candlestick', verbose)
    
    def _compare_prediction_plots(self, python_plots, matlab_plots, verbose=False):
        """Compare prediction plots"""
        python_pred = [p for p in python_plots if 'prediction' in p.lower()]
        matlab_pred = [m for m in matlab_plots if 'prediction' in m.lower() or 'predicted' in m.lower()]
        
        if verbose:
            print(f"Python prediction plots: {python_pred}")
            print(f"MATLAB prediction plots: {matlab_pred}")
        
        if not python_pred or not matlab_pred:
            return {'status': 'no_plots', 'message': 'No prediction plots found for comparison'}
        
        # Compare the first available plots
        python_plot = python_pred[0]
        matlab_plot = matlab_pred[0]
        
        return self._compare_plot_files(python_plot, matlab_plot, 'prediction', verbose)
    
    def _compare_plot_files(self, python_file, matlab_file, plot_type, verbose=False):
        """Compare two plot files"""
        try:
            # Load images
            python_img = mpimg.imread(python_file)
            matlab_img = mpimg.imread(matlab_file)
            
            # Basic comparison
            comparison = {
                'plot_type': plot_type,
                'python_file': python_file,
                'matlab_file': matlab_file,
                'python_shape': python_img.shape,
                'matlab_shape': matlab_img.shape,
                'timestamp': datetime.now().isoformat()
            }
            
            # Size comparison
            if python_img.shape == matlab_img.shape:
                comparison['size_match'] = True
                comparison['size_difference'] = 0
            else:
                comparison['size_match'] = False
                comparison['size_difference'] = abs(python_img.size - matlab_img.size)
            
            # Basic statistics
            comparison['python_mean'] = float(np.mean(python_img))
            comparison['matlab_mean'] = float(np.mean(matlab_img))
            comparison['python_std'] = float(np.std(python_img))
            comparison['matlab_std'] = float(np.std(matlab_img))
            
            # Calculate difference metrics
            if python_img.shape == matlab_img.shape:
                diff = np.abs(python_img - matlab_img)
                comparison['mean_absolute_difference'] = float(np.mean(diff))
                comparison['max_difference'] = float(np.max(diff))
                comparison['similarity_score'] = float(1.0 - np.mean(diff))
            else:
                comparison['mean_absolute_difference'] = None
                comparison['max_difference'] = None
                comparison['similarity_score'] = None
            
            comparison['status'] = 'success'
            
            if verbose:
                print(f"Comparison completed for {plot_type}")
                print(f"   Python: {python_file} ({python_img.shape})")
                print(f"   MATLAB: {matlab_file} ({matlab_img.shape})")
                print(f"   Similarity: {comparison['similarity_score']:.4f}")
            
            return comparison
            
        except Exception as e:
            return {
                'plot_type': plot_type,
                'python_file': python_file,
                'matlab_file': matlab_file,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _save_plot_results(self, results):
        """Save plot comparison results to JSON"""
        output_file = os.path.join(self.results_dir, 'plot_comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Plot comparison results saved to: {output_file}")