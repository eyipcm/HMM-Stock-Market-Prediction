"""
Report generator for HMM Stock Market Prediction comparison
Generates comprehensive comparison reports in Markdown and JSON formats

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
import json
from datetime import datetime

class ReportGenerator:
    def __init__(self, results_dir):
        self.results_dir = results_dir
        
    def generate_comprehensive_report(self, plot_results, data_results, image_results):
        """Generate a comprehensive comparison report"""
        print("Generating comprehensive report...")
        
        # Generate Markdown report
        self._generate_markdown_report(plot_results, data_results, image_results)
        
        # Generate JSON report
        self._generate_json_report(plot_results, data_results, image_results)
        
        print("Comprehensive report generated successfully!")
    
    def _generate_markdown_report(self, plot_results, data_results, image_results):
        """Generate Markdown report"""
        report_file = os.path.join(self.results_dir, 'comparison_report.md')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# HMM Stock Market Prediction - Python vs MATLAB Comparison Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("This report compares the Python implementation of the HMM Stock Market Prediction system with the original MATLAB implementation.\n\n")
            
            # Plot Comparison Results
            f.write("## Plot Comparison Results\n\n")
            if plot_results:
                for plot_type, result in plot_results.items():
                    f.write(f"### {plot_type.title()} Plots\n\n")
                    if result.get('status') == 'success':
                        f.write(f"- **Status:** [SUCCESS]\n")
                        f.write(f"- **Python File:** `{result.get('python_file', 'N/A')}`\n")
                        f.write(f"- **MATLAB File:** `{result.get('matlab_file', 'N/A')}`\n")
                        f.write(f"- **Size Match:** {'Yes' if result.get('size_match') else 'No'}\n")
                        f.write(f"- **Similarity Score:** {result.get('similarity_score', 'N/A'):.4f}\n")
                        f.write(f"- **Mean Absolute Difference:** {result.get('mean_absolute_difference', 'N/A')}\n")
                    else:
                        f.write(f"- **Status:** [ERROR] {result.get('status', 'Unknown')}\n")
                        f.write(f"- **Message:** {result.get('message', 'No message available')}\n")
                    f.write("\n")
            else:
                f.write("No plot comparison results available.\n\n")
            
            # Data Comparison Results
            f.write("## Data Comparison Results\n\n")
            if data_results:
                for data_type, result in data_results.items():
                    f.write(f"### {data_type.title()} Data\n\n")
                    f.write(f"- **Status:** {result.get('status', 'Unknown')}\n")
                    f.write(f"- **Message:** {result.get('message', 'No message available')}\n")
                    f.write("\n")
            else:
                f.write("No data comparison results available.\n\n")
            
            # Image Comparison Results
            f.write("## Image Comparison Results\n\n")
            if image_results:
                for comparison_name, result in image_results.items():
                    f.write(f"### {comparison_name}\n\n")
                    if result.get('status') == 'success':
                        f.write(f"- **Status:** [SUCCESS]\n")
                        f.write(f"- **MSE:** {result.get('mse', 'N/A'):.4f}\n")
                        f.write(f"- **PSNR:** {result.get('psnr', 'N/A'):.4f}\n")
                        f.write(f"- **SSIM:** {result.get('ssim', 'N/A'):.4f}\n")
                        f.write(f"- **Overall Similarity:** {result.get('overall_similarity', 'N/A'):.4f}\n")
                    else:
                        f.write(f"- **Status:** [ERROR] {result.get('status', 'Unknown')}\n")
                        f.write(f"- **Error:** {result.get('error', 'No error message')}\n")
                    f.write("\n")
            else:
                f.write("No image comparison results available.\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("1. **Plot Quality:** Ensure both implementations generate plots with similar dimensions and quality\n")
            f.write("2. **Data Consistency:** Verify that both implementations produce consistent numerical results\n")
            f.write("3. **Performance:** Compare execution times and memory usage between implementations\n")
            f.write("4. **Documentation:** Maintain clear documentation of differences between implementations\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            f.write("This comparison provides a baseline for understanding the differences between the Python and MATLAB implementations. Use these results to guide further development and ensure consistency between the two versions.\n")
        
        print(f"Markdown report saved to: {report_file}")
    
    def _generate_json_report(self, plot_results, data_results, image_results):
        """Generate JSON report"""
        report_file = os.path.join(self.results_dir, 'comparison_results.json')
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'comprehensive_comparison',
                'version': '1.0'
            },
            'plot_comparison': plot_results,
            'data_comparison': data_results,
            'image_comparison': image_results,
            'summary': {
                'total_plot_comparisons': len(plot_results) if plot_results else 0,
                'total_data_comparisons': len(data_results) if data_results else 0,
                'total_image_comparisons': len(image_results) if image_results else 0
            }
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"JSON report saved to: {report_file}")