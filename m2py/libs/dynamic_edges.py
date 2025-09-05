#!/usr/bin/env python3
"""
Dynamic edges calculation for discretization
Converted from MATLAB libs/dynamicEdges.m
"""

import numpy as np

def dynamic_edges(sequence, number_of_points):
    """
    Calculate dynamic intervals for discretization of a sequence.
    
    Args:
        sequence: The sequence of real values to discretize
        number_of_points: The desired number of points for discretization
    
    Returns:
        edges: The edges of the intervals calculated for discretization
    """
    if not isinstance(sequence, (list, np.ndarray)):
        raise TypeError('sequence must be a vector')
    
    if not isinstance(number_of_points, (int, np.integer)):
        raise TypeError('number_of_points must be a scalar')
    
    sequence = np.array(sequence)
    min_seq = np.min(sequence)
    max_seq = np.max(sequence)
    
    if min_seq < -0.9 or max_seq > 0.9:
        print("Warning: Weird min/max value(s)")
    
    edges = np.linspace(min_seq, max_seq, number_of_points + 1)
    return edges