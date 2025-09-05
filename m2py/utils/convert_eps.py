"""
Convert MATLAB EPS files to PNG for comparison
"""

import os
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
import numpy as np

def convert_eps_to_png(eps_dir, png_dir):
    """Convert all EPS files in a directory to PNG"""
    os.makedirs(png_dir, exist_ok=True)
    
    converted_files = []
    
    for file in os.listdir(eps_dir):
        if file.endswith('.eps'):
            eps_path = os.path.join(eps_dir, file)
            png_file = file.replace('.eps', '.png')
            png_path = os.path.join(png_dir, png_file)
            
            try:
                # Read EPS file
                img = mpimg.imread(eps_path)
                
                # Save as PNG
                plt.imsave(png_path, img)
                converted_files.append(png_path)
                print(f"Converted: {file} -> {png_file}")
                
            except Exception as e:
                print(f"Failed to convert {file}: {e}")
    
    return converted_files

if __name__ == "__main__":
    # Convert MATLAB EPS files to PNG
    eps_dir = "../../out_figs"
    png_dir = "../../out_figs_png"
    
    print("Converting MATLAB EPS files to PNG...")
    converted = convert_eps_to_png(eps_dir, png_dir)
    print(f"Converted {len(converted)} files")