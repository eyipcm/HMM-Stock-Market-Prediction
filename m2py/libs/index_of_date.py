#!/usr/bin/env python3
"""
Find index of target date in dates array
Converted from MATLAB libs/indexOfDate.m

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

import pandas as pd

def index_of_date(dates, target_date):
    """
    Find the index in the dates array of the target date.
    
    Args:
        dates: pandas Series or array of dates
        target_date: target date string in 'YYYY-MM-DD' format
    
    Returns:
        int: index of the target date
    """
    target_dt = pd.to_datetime(target_date)
    matches = dates == target_dt
    
    if not matches.any():
        raise ValueError(f'Date {target_date} not found')
    
    return matches.idxmax()