#!/usr/bin/env python3
"""
Map 1D index back to 3D coordinates
Converted from MATLAB libs/map/map1DTo3D.m
"""

import numpy as np

def map_1d_to_3d(n, max_x, max_y, max_z):
    """
    Perform inverse mapping, converting an integer n to the corresponding 
    triple (x, y, z) in 3D space.
    
    Args:
        n: The integer assigned to the triple (x, y, z) in 3D space mapped to 1D
        max_x, max_y, max_z: The maximum values of x, y, z in 3D space
    
    Returns:
        tuple: (x, y, z) coordinates corresponding to the 1D index in 3D space
    """
    max_possible = (max_z - 1) * (max_x * max_y) + (max_y - 1) * max_x + max_x
    
    if n > max_possible:
        raise ValueError(f"Invalid number to convert n = {n:.4f}")
    
    z = int((n - 1) // (max_x * max_y)) + 1
    y = int(((n - 1) - (z - 1) * (max_x * max_y)) // max_x) + 1
    x = int((n - 1) % max_x) + 1
    
    return x, y, z