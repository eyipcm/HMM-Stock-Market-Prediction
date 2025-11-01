#!/usr/bin/env python3
"""
Main comparison script for HMM Stock Market Prediction
Compares Python implementation outputs with MATLAB implementation outputs

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
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plot_comparison import PlotComparison
from utils.data_comparison import DataComparison
from utils.image_comparison import ImageComparison
from utils.report_generator import ReportGenerator

def main():
    """Main comparison function"""
    parser = argparse.ArgumentParser(description='Compare Python and MATLAB HMM implementations')
    parser.add_argument('--python-output', default='../output_figs', 
                       help='Directory containing Python outputs')
    parser.add_argument('--matlab-output', default='../../out_figs', 
                       help='Directory containing MATLAB outputs')
    parser.add_argument('--output-dir', default='comparison_results', 
                       help='Directory to save comparison results')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HMM Stock Market Prediction - Python vs MATLAB Comparison")
    print("=" * 60)
    print(f"Python outputs: {args.python_output}")
    print(f"MATLAB outputs: {args.matlab_output}")
    print(f"Results directory: {args.output_dir}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize comparison tools
        plot_comp = PlotComparison(args.python_output, args.matlab_output, args.output_dir)
        data_comp = DataComparison(args.python_output, args.matlab_output, args.output_dir)
        image_comp = ImageComparison(args.python_output, args.matlab_output, args.output_dir)
        report_gen = ReportGenerator(args.output_dir)
        
        # Run comparisons
        print("\n1. Comparing plots...")
        plot_results = plot_comp.compare_all_plots(verbose=args.verbose)
        
        print("\n2. Comparing data...")
        data_results = data_comp.compare_all_data(verbose=args.verbose)
        
        print("\n3. Comparing images...")
        image_results = image_comp.compare_all_images(verbose=args.verbose)
        
        # Generate comprehensive report
        print("\n4. Generating comparison report...")
        report_gen.generate_comprehensive_report(plot_results, data_results, image_results)
        
        print(f"\nComparison completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report: {args.output_dir}/comparison_report.md")
        print(f"JSON: {args.output_dir}/comparison_results.json")
        
    except Exception as e:
        print(f"Error during comparison: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()