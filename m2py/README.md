# HMM Stock Market Prediction - Python Implementation

This is the Python conversion of the MATLAB HMM Stock Market Prediction project. The implementation uses Hidden Markov Models (HMMs) to predict stock closing prices based on historical data.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Output](#output)
- [MATLAB Output Conversion](#matlab-output-conversion)
- [Key Differences from MATLAB](#key-differences-from-matlab)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## 🔍 Overview

The project implements a Hidden Markov Model for stock market prediction using:
- **Fractional changes** in stock prices (Open, High, Low, Close)
- **Discretization** of continuous observations into discrete states
- **Gaussian Mixture Models** for emission probabilities
- **HMM training** using the Baum-Welch algorithm
- **Prediction** of future stock prices

## ✨ Features

- **Stock Data Processing**: Load and process CSV stock data
- **HMM Training**: Train custom HMM models with configurable parameters
- **Price Prediction**: Predict future stock closing prices
- **Visualization**: Generate candlestick charts and prediction plots
- **Performance Metrics**: Calculate MAPE and DPA (Directional Prediction Accuracy)
- **Model Persistence**: Save and load trained models
- **Flexible Configuration**: Easy parameter adjustment

## 📦 Requirements

### Python Version
- Python 3.11 or higher

### Required Packages
```
cffi==1.17.1
contourpy==1.3.3
cycler==0.12.1
fonttools==4.59.2
hmmlearn==0.3.3
imageio==2.37.0
joblib==1.5.2
kiwisolver==1.4.9
lazy_loader==0.4
matplotlib==3.10.6
mplfinance==0.12.10b0
narwhals==2.3.0
networkx==3.5
numpy==2.2.6
opencv-python==4.12.0.88
packaging==25.0
pandas==2.3.2
pillow==11.3.0
plotly==6.3.0
pycparser==2.22
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
scikit-image==0.25.2
scikit-learn==1.7.1
scipy==1.16.1
seaborn==0.13.2
six==1.17.0
sounddevice==0.5.2
soundfile==0.13.1
threadpoolctl==3.6.0
tifffile==2025.8.28
tzdata==2025.2
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd HMM-Stock-Market-Prediction
   ```

2. **Navigate to Python implementation:**
   ```bash
   cd m2py
   ```

3. **Create a virtual environment (Recommended):**

   > **Why use a virtual environment?**
   > - Prevents dependency conflicts with other Python projects
   > - Keeps your system Python clean
   > - Ensures reproducible installations
   > - Makes it easier to manage different project requirements

   **Option A: Using venv (Python built-in):**
   ```bash
   # Create virtual environment
   python -m venv hmm_env
   
   # Activate virtual environment
   # On Windows:
   hmm_env\Scripts\activate
   # On macOS/Linux:
   source hmm_env/bin/activate
   ```

   **Option B: Using conda:**
   ```bash
   # Create conda environment
   conda create -n hmm_env python=3.11
   
   # Activate conda environment
   conda activate hmm_env
   ```

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ensure data files are available:**
   - Place CSV files in `../datasets/csv/` directory
   - Required files: `AAPL.csv`, `DELL.csv`, `IBM.csv`, `VIX.csv`

6. **Verify installation:**
   ```bash
   python -c "import numpy, pandas, matplotlib, hmmlearn; print('All packages installed successfully!')"
   ```

### Virtual Environment Management

**To deactivate the virtual environment:**
```bash
# For venv:
deactivate

# For conda:
conda deactivate
```

**To reactivate the environment later:**
```bash
# For venv:
# On Windows:
hmm_env\Scripts\activate
# On macOS/Linux:
source hmm_env/bin/activate

# For conda:
conda activate hmm_env
```

**To remove the virtual environment:**
```bash
# For venv:
# Simply delete the hmm_env folder
rm -rf hmm_env  # On macOS/Linux
rmdir /s hmm_env  # On Windows

# For conda:
conda env remove -n hmm_env
```

## 💻 Usage

### Basic Usage

1. **Run the main script:**
   ```bash
   python main.py
   ```

2. **For training a new model:**
   - Edit `init_config.py`
   - Set `'TRAIN': 1`
   - Run `python main.py`

3. **For prediction with existing model:**
   - Edit `init_config.py`
   - Set `'TRAIN': 0`
   - Update `'filename'` to point to your model
   - Run `python main.py`

### Configuration

Edit `init_config.py` to customize:

```python
config = {
    'stock_name': 'AAPL.csv',           # Stock data file
    'TRAIN': 0,                         # 0: load model, 1: train new
    'filename': 'train/hmmtrain-20230717025206.pkl',  # Model file
    'start_train_date': '2017-01-03',   # Training start date
    'end_train_date': '2019-01-03',     # Training end date
    'start_prediction_date': '2023-01-03',  # Prediction start date
    'underlying_states': 4,             # Number of hidden states
    'mixtures_number': 4,               # Gaussian mixture components
    'latency': 10,                      # Sequence length
    'discretization_points': [50, 10, 10],  # Discretization bins
    'use_dynamic_edges': 0,             # Dynamic vs fixed discretization
}
```

## 📁 File Structure

```
m2py/
├── main.py                    # Main execution script
├── init_config.py            # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── train/                    # Trained HMM models (.pkl files)
│   └── hmmtrain-YYYYMMDDhhmmss.pkl
├── output_figs/             # Generated plots (.png files)
│   ├── AAPL_candlestick_YYYYMMDDhhmmss.png
│   └── AAPL_predictions_YYYYMMDDhhmmss.png
└── libs/                    # Library functions
    ├── __init__.py
    ├── dynamic_edges.py     # Dynamic discretization
    ├── index_of_date.py     # Date utilities
    ├── hmm_predict_observation.py  # HMM prediction
    ├── rg_candle.py         # Candlestick plotting
    └── map/                 # Coordinate mapping
        ├── __init__.py
        ├── map_3d_to_1d.py  # 3D to 1D mapping
        └── map_1d_to_3d.py  # 1D to 3D mapping
```

## 📊 Output

### Console Output
- Training progress and status
- Prediction progress (percentage complete)
- Performance metrics (MAPE and DPA)
- File save locations

### Files Generated
- **Models**: `train/hmmtrain-YYYYMMDDhhmmss.pkl`
- **Plots**: `output_figs/STOCK_candlestick_YYYYMMDDhhmmss.png`
- **Plots**: `output_figs/STOCK_predictions_YYYYMMDDhhmmss.png`

### Performance Metrics
- **MAPE**: Mean Absolute Percentage Error
- **DPA**: Directional Prediction Accuracy (correct direction predictions)

## 🔄 MATLAB Output Conversion

### Converting MATLAB .eps Files to PNG

The MATLAB implementation generates `.eps` (Encapsulated PostScript) files in the `out_figs/` directory. To convert these to PNG format for easier viewing and comparison:

#### Method 1: Using Python Script (Recommended)

1. **Run the conversion script:**
   ```bash
   cd m2py/utils
   python convert_eps.py
   ```

2. **The script will:**
   - Convert all `.eps` files from `../../out_figs/` to PNG format
   - Save converted files to `../../out_figs_png/`
   - Maintain original filenames with `.png` extension

#### Method 2: Manual Conversion

1. **Using ImageMagick (if installed):**
   ```bash
   # Convert single file
   magick convert input.eps output.png
   
   # Convert all .eps files in directory
   for file in ../../out_figs/*.eps; do
       magick convert "$file" "../../out_figs_png/$(basename "$file" .eps).png"
   done
   ```

2. **Using Ghostscript:**
   ```bash
   # Convert single file
   gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r300 -sOutputFile=output.png input.eps
   
   # Convert all .eps files
   for file in ../../out_figs/*.eps; do
       gs -dNOPAUSE -dBATCH -sDEVICE=png16m -r300 -sOutputFile="../../out_figs_png/$(basename "$file" .eps).png" "$file"
   done
   ```

#### Method 3: Using MATLAB

If you have MATLAB available, you can use the built-in conversion:

```matlab
% Navigate to the out_figs directory
cd out_figs

% Get all .eps files
eps_files = dir('*.eps');

% Convert each file
for i = 1:length(eps_files)
    [~, name, ~] = fileparts(eps_files(i).name);
    eps_file = fullfile(eps_files(i).folder, eps_files(i).name);
    png_file = fullfile('../out_figs_png', [name '.png']);
    
    % Read and convert
    [img, map] = imread(eps_file);
    imwrite(img, map, png_file);
end
```

### Directory Structure After Conversion

```
HMM-Stock-Market-Prediction/
├── out_figs/                    # Original MATLAB .eps files
│   ├── AAPL 2022-end candlestick.eps
│   ├── AAPL hmmtrain-2023-07-15-17-31-41 candlestick.eps
│   ├── AAPL hmmtrain-2023-07-15-17-31-41 DPA.eps
│   └── ...
├── out_figs_png/               # Converted PNG files
│   ├── AAPL 2022-end candlestick.png
│   ├── AAPL hmmtrain-2023-07-15-17-31-41 candlestick.png
│   ├── AAPL hmmtrain-2023-07-15-17-31-41 DPA.png
│   └── ...
└── m2py/
    ├── output_figs/            # Python generated PNG files
    │   ├── AAPL_candlestick_YYYYMMDDhhmmss.png
    │   └── AAPL_predictions_YYYYMMDDhhmmss.png
    └── ...
```

### Comparison Tools

The project includes comparison utilities to analyze differences between MATLAB and Python outputs:

1. **Run comparison analysis:**
   ```bash
   cd m2py/utils
   python comparison_main.py --python-output ../../m2py/output_figs --matlab-output ../../out_figs_png --verbose
   ```

2. **Generate comparison report:**
   - Creates detailed comparison in `comparison_results/`
   - Includes image differences, data analysis, and plot comparisons
   - Generates markdown report with findings

### Required Dependencies for Conversion

The conversion process uses packages already included in the main requirements:
- **Pillow** (11.3.0) - For image processing
- **matplotlib** (3.10.6) - For EPS reading
- **numpy** (2.2.6) - For array operations
- **opencv-python** (4.12.0.88) - For advanced image processing
- **scikit-image** (0.25.2) - For image analysis

## 🔄 Key Differences from MATLAB

| Aspect | MATLAB | Python |
|--------|--------|--------|
| **Data Format** | .mat files | CSV files |
| **HMM Library** | Built-in HMM functions | hmmlearn |
| **Plotting** | MATLAB plotting | matplotlib |
| **File I/O** | .mat save/load | pickle |
| **Indexing** | 1-based | 0-based (with conversion) |
| **Model Files** | .mat | .pkl |
| **Timestamp** | YYYY-MM-DD-HH-MM-SS | YYYYMMDDhhmmss |

## 📈 Example Results

```
Init
Markov Chain guesses
Training HMM...
Model saved to train/hmmtrain-20231215143025.pkl
Training time: 45.67 seconds
Making predictions...
25.00% : 50.00% : 75.00% : 100.00% : 

Mean Absolute Percentage Error (MAPE): 2.34%
Directional Prediction Accuracy (DPA): 67.89%
Analysis complete!
```

## 📄 License

See [LICENSE](LICENSE) file.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the original MATLAB implementation
- Create an issue in the repository

---

**Note**: This Python implementation maintains the same functionality as the original MATLAB code while providing better portability and integration with modern data science workflows.
