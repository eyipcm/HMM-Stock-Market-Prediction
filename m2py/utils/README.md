# HMM Stock Market Prediction - Comparison Utilities

This directory contains comprehensive comparison tools to evaluate the Python implementation against the original MATLAB implementation of the HMM Stock Market Prediction system.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Comparison Methods](#comparison-methods)
- [Output Formats](#output-formats)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🔍 Overview

The comparison utilities provide systematic analysis of:
- **Visual Quality**: Compare candlestick plots and prediction visualizations
- **Data Consistency**: Verify data accuracy between implementations
- **Performance Metrics**: Quantitative comparison of outputs
- **Comprehensive Reporting**: Generate detailed comparison reports

## ✨ Features

- **Plot Comparison**: Side-by-side and overlay visual comparisons
- **Data Analysis**: Statistical comparison of OHLC data
- **Image Metrics**: Quantitative image similarity analysis
- **Report Generation**: Automated markdown and JSON reports
- **Flexible Configuration**: Customizable comparison parameters
- **Batch Processing**: Compare multiple outputs simultaneously

## 📁 File Structure

```
utils/
├── __init__.py              # Package initialization
├── plot_comparison.py       # Candlestick plot comparison
├── data_comparison.py       # Data consistency analysis
├── image_comparison.py      # Image similarity metrics
├── report_generator.py      # Report generation utilities
├── comparison_main.py       # Main comparison script
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- Required packages (installed with main project):
  - matplotlib
  - numpy
  - pandas
  - scipy
  - PIL (Pillow)

### Setup
```bash
# Navigate to utils directory
cd m2py/utils

# Ensure parent packages are available
python -c "import sys; sys.path.append('..'); from libs import *"
```

## 💻 Usage

### Quick Start

1. **Run comprehensive comparison:**
   ```bash
   python comparison_main.py
   ```

2. **Run individual comparisons:**
   ```bash
   # Plot comparison only
   python -c "from plot_comparison import compare_candlestick_plots; compare_candlestick_plots()"
   
   # Data comparison only
   python -c "from data_comparison import compare_data_consistency; compare_data_consistency()"
   ```

### Advanced Usage

#### Plot Comparison
```python
from plot_comparison import compare_candlestick_plots

# Compare specific files
metrics = compare_candlestick_plots(
    python_fig_path='../output_figs/AAPL_candlestick_20231215143025.png',
    matlab_fig_path='../../out_figs/AAPL hmmtrain-2023-07-15-17-31-41 candlestick.eps',
    output_dir='my_comparison'
)
```

#### Data Analysis
```python
from data_comparison import compare_data_consistency

# Compare data with custom date range
compare_data_consistency(
    python_data_path='../datasets/csv/AAPL.csv',
    start_date='2023-01-03',
    end_date='2023-06-30'
)
```

#### Image Metrics
```python
from image_comparison import compare_images

# Compare two images
metrics = compare_images(
    img1_path='python_plot.png',
    img2_path='matlab_plot.png'
)
```

## 🔬 Comparison Methods

### 1. Visual Comparison
- **Side-by-Side**: Display Python and MATLAB plots adjacent
- **Overlay**: Superimpose plots for direct comparison
- **Difference Image**: Highlight visual differences

### 2. Quantitative Metrics
- **MSE (Mean Squared Error)**: Pixel-level difference measurement
- **PSNR (Peak Signal-to-Noise Ratio)**: Image quality metric
- **SSIM (Structural Similarity Index)**: Perceptual similarity
- **Histogram Correlation**: Distribution similarity
- **Edge Similarity**: Feature-level comparison

### 3. Data Consistency
- **OHLC Comparison**: Open, High, Low, Close price differences
- **Date Range Verification**: Ensure same time periods
- **Statistical Analysis**: Mean, std, min, max differences
- **Missing Data Detection**: Identify data gaps

## 📊 Output Formats

### Generated Files
```
comparison_output_YYYYMMDD_HHMMSS/
├── candlestick_comparison_YYYYMMDDhhmmss.png    # Side-by-side plot
├── candlestick_overlay_YYYYMMDDhhmmss.png       # Overlay comparison
├── image_comparison_YYYYMMDDhhmmss.png          # Detailed image analysis
├── comparison_report_YYYYMMDDhhmmss.md         # Markdown report
└── comparison_data_YYYYMMDDhhmmss.json          # JSON data
```

### Report Contents
- **Executive Summary**: Key findings and recommendations
- **Visual Analysis**: Plot comparison results
- **Data Verification**: Consistency checks
- **Performance Metrics**: Quantitative comparisons
- **Recommendations**: Improvement suggestions

## 📈 Examples

### Example Output
```
=== Plot Comparison Metrics ===
Mean Squared Error: 0.0234
Peak Signal-to-Noise Ratio: 45.67 dB
Histogram Correlation: 0.9876
Python Image Shape: (800, 1200, 3)
MATLAB Image Shape: (800, 1200, 3)

=== Data Differences Analysis ===
Open Differences:
  Mean: 0.000123
  Max: 0.001234
  Std: 0.000456
  Zero differences: 245/250

Comparison report generated:
  Markdown: comparison_output_20231215_143025/comparison_report_20231215143025.md
  JSON: comparison_output_20231215_143025/comparison_data_20231215143025.json
```

### Sample Report Structure
```markdown
# HMM Stock Market Prediction - Comparison Report

## Overview
- Python implementation shows 98.76% visual similarity
- Data consistency verified with <0.01% differences
- Performance metrics within acceptable ranges

## Recommendations
1. Visual Quality: Excellent match with MATLAB output
2. Data Accuracy: High consistency confirmed
3. Performance: Python implementation 15% faster
4. Improvements: Consider additional plot customization options
```

## 🔧 Configuration

### Customizing Comparisons

#### Plot Comparison Parameters
```python
# In plot_comparison.py
def compare_candlestick_plots(
    python_fig_path=None,      # Auto-detect if None
    matlab_fig_path=None,      # Auto-detect if None
    output_dir='comparison_output',  # Output directory
    dpi=300,                   # Image resolution
    figsize=(20, 10)          # Figure size
):
```

#### Data Comparison Parameters
```python
# In data_comparison.py
def compare_data_consistency(
    python_data_path='../datasets/csv/AAPL.csv',
    matlab_data_path=None,
    start_date='2023-01-03',   # Comparison start
    end_date='2023-06-30',     # Comparison end
    tolerance=1e-6             # Difference tolerance
):
```

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Ensure you're in the utils directory
   cd m2py/utils
   
   # Add parent to path
   python -c "import sys; sys.path.append('..'); from plot_comparison import *"
   ```

2. **File Not Found:**
   ```bash
   # Check file paths
   dir ..\output_figs\*.png
   dir ..\..\out_figs\*.eps
   ```

3. **Image Loading Errors:**
   ```bash
   # Install additional dependencies
   pip install Pillow
   pip install scikit-image
   ```

4. **Memory Issues:**
   ```python
   # Reduce image resolution
   dpi=150  # Instead of 300
   ```

### Performance Tips

- **Faster Processing**: Use lower DPI settings
- **Memory Efficient**: Process one comparison at a time
- **Batch Processing**: Use wildcards for multiple files

## 📝 Contributing

### Adding New Comparison Methods

1. **Create new comparison function:**
   ```python
   def new_comparison_method(data1, data2):
       """New comparison method"""
       # Implementation
       return metrics
   ```

2. **Update __init__.py:**
   ```python
   from .new_comparison import new_comparison_method
   __all__.append('new_comparison_method')
   ```

3. **Add to main script:**
   ```python
   # In comparison_main.py
   from new_comparison import new_comparison_method
   results['new_metrics'] = new_comparison_method(data1, data2)
   ```

### Reporting Issues

When reporting issues, include:
- Python version
- Package versions
- Error messages
- File paths
- Expected vs actual behavior

## 📄 License

This utilities package follows the same license as the main project. See [LICENSE](../../LICENSE.md).

## 🤝 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages carefully
3. Verify file paths and permissions
4. Create an issue in the repository

---

**Note**: These utilities are designed to provide comprehensive comparison between Python and MATLAB implementations, ensuring the Python conversion maintains the same quality and accuracy as the original MATLAB code.
```

## **To Create the File:**

1. Open a text editor (Notepad, VS Code, etc.)
2. Copy the entire content above
3. Save it as `README.md` in the `D:\gitrepo\HMM-Stock-Market-Prediction\m2py\utils\` directory
4. Make sure the file encoding is UTF-8

This README provides comprehensive documentation for the comparison utilities, including usage examples, configuration options, troubleshooting tips, and contribution guidelines.



