#!/usr/bin/env python3
"""
Map 3D coordinates to 1D index
Converted from MATLAB libs/map/map3DTo1D.m
"""

import numpy as np

def map_3d_to_1d(x, y, z, max_x, max_y, max_z):
    """
    Map 3D space coordinates to a single dimension.
    
    Args:
        x, y, z: The coordinates of the triple (x, y, z) in 3D space
        max_x, max_y, max_z: The maximum values of x, y, z in 3D space
    
    Returns:
        int: The integer assigned to the triple (x, y, z) in 3D space mapped to 1D
    """
    if z > max_z or y > max_y or x > max_x:
        raise ValueError(f"Invalid triple to convert: x = {x:.4f}, y = {y:.4f}, z = {z:.4f}")
    
    n = (z - 1) * (max_x * max_y) + (y - 1) * max_x + x
    
    if np.isnan(n):
        raise ValueError(f"Mapped to NaN. x = {x:.2f}; y = {y:.2f}; z = {z:.2f}")
    
    return int(n)