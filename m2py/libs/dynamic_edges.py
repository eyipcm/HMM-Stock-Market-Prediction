#!/usr/bin/env python3
"""
Dynamic edges calculation for discretization
Converted from MATLAB libs/dynamicEdges.m

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