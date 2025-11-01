"""
Image comparison utilities for HMM Stock Market Prediction
Provides quantitative image comparison metrics

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
from PIL import Image
import json
from datetime import datetime

class ImageComparison:
    def __init__(self, python_output_dir, matlab_output_dir, results_dir):
        self.python_output_dir = python_output_dir
        self.matlab_output_dir = matlab_output_dir
        self.results_dir = results_dir
        self.comparison_results = {}
        
    def compare_all_images(self, verbose=False):
        """Compare all available images between Python and MATLAB outputs"""
        print("Comparing images...")
        
        # Find all image files
        python_images = self._find_image_files(self.python_output_dir)
        matlab_images = self._find_image_files(self.matlab_output_dir)
        
        if verbose:
            print(f"Found {len(python_images)} Python images")
            print(f"Found {len(matlab_images)} MATLAB images")
        
        results = {}
        
        # Compare each image pair
        for python_img in python_images:
            for matlab_img in matlab_images:
                if self._is_similar_image(python_img, matlab_img):
                    comparison = self._compare_image_pair(python_img, matlab_img, verbose)
                    results[f"{os.path.basename(python_img)}_vs_{os.path.basename(matlab_img)}"] = comparison
        
        # Save results
        self.comparison_results = results
        self._save_image_results(results)
        
        return results
    
    def _find_image_files(self, directory):
        """Find all image files in a directory"""
        image_files = []
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_files.append(os.path.join(directory, file))
        return image_files
    
    def _is_similar_image(self, python_img, matlab_img):
        """Check if two images are likely to be comparable"""
        python_name = os.path.basename(python_img).lower()
        matlab_name = os.path.basename(matlab_img).lower()
        
        # Simple heuristic: check if they contain similar keywords
        keywords = ['candle', 'prediction', 'plot', 'chart']
        for keyword in keywords:
            if keyword in python_name and keyword in matlab_name:
                return True
        return False
    
    def _compare_image_pair(self, python_img, matlab_img, verbose=False):
        """Compare a pair of images using various metrics"""
        try:
            # Load images
            img1 = Image.open(python_img).convert('RGB')
            img2 = Image.open(matlab_img).convert('RGB')
            
            # Convert to numpy arrays
            arr1 = np.array(img1)
            arr2 = np.array(img2)
            
            # Resize to same dimensions if needed
            if arr1.shape != arr2.shape:
                # Resize to smaller dimensions
                min_height = min(arr1.shape[0], arr2.shape[0])
                min_width = min(arr1.shape[1], arr2.shape[1])
                arr1 = arr1[:min_height, :min_width]
                arr2 = arr2[:min_height, :min_width]
            
            # Calculate comparison metrics
            comparison = {
                'python_file': python_img,
                'matlab_file': matlab_img,
                'timestamp': datetime.now().isoformat()
            }
            
            # Basic metrics
            comparison['mse'] = float(np.mean((arr1 - arr2) ** 2))
            comparison['psnr'] = float(self._calculate_psnr(arr1, arr2))
            comparison['ssim'] = float(self._calculate_ssim(arr1, arr2))
            comparison['histogram_correlation'] = float(self._calculate_histogram_correlation(arr1, arr2))
            comparison['edge_similarity'] = float(self._calculate_edge_similarity(arr1, arr2))
            
            # Overall similarity score
            comparison['overall_similarity'] = float(
                (comparison['ssim'] + comparison['histogram_correlation'] + comparison['edge_similarity']) / 3
            )
            
            comparison['status'] = 'success'
            
            if verbose:
                print(f"✅ Image comparison completed")
                print(f"   MSE: {comparison['mse']:.4f}")
                print(f"   PSNR: {comparison['psnr']:.4f}")
                print(f"   SSIM: {comparison['ssim']:.4f}")
                print(f"   Overall similarity: {comparison['overall_similarity']:.4f}")
            
            return comparison
            
        except Exception as e:
            return {
                'python_file': python_img,
                'matlab_file': matlab_img,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_psnr(self, img1, img2):
        """Calculate Peak Signal-to-Noise Ratio"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def _calculate_ssim(self, img1, img2):
        """Calculate Structural Similarity Index"""
        # Simplified SSIM calculation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.var(img1)
        sigma2 = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        return ssim
    
    def _calculate_histogram_correlation(self, img1, img2):
        """Calculate histogram correlation"""
        hist1 = np.histogram(img1.flatten(), bins=256, range=(0, 256))[0]
        hist2 = np.histogram(img2.flatten(), bins=256, range=(0, 256))[0]
        
        # Normalize histograms
        hist1 = hist1 / np.sum(hist1)
        hist2 = hist2 / np.sum(hist2)
        
        # Calculate correlation
        correlation = np.corrcoef(hist1, hist2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
    
    def _calculate_edge_similarity(self, img1, img2):
        """Calculate edge similarity using simple edge detection"""
        # Convert to grayscale
        gray1 = np.mean(img1, axis=2)
        gray2 = np.mean(img2, axis=2)
        
        # Simple edge detection using gradient
        edges1 = np.abs(np.gradient(gray1)[0]) + np.abs(np.gradient(gray1)[1])
        edges2 = np.abs(np.gradient(gray2)[0]) + np.abs(np.gradient(gray2)[1])
        
        # Normalize
        edges1 = edges1 / np.max(edges1)
        edges2 = edges2 / np.max(edges2)
        
        # Calculate similarity
        similarity = 1.0 - np.mean(np.abs(edges1 - edges2))
        return similarity
    
    def _save_image_results(self, results):
        """Save image comparison results to JSON"""
        output_file = os.path.join(self.results_dir, 'image_comparison_results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Image comparison results saved to: {output_file}")