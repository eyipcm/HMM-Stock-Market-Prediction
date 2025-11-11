#!/usr/bin/env python3
"""
Mapping utilities for 3D to 1D coordinate conversion

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

from .map_3d_to_1d import map_3d_to_1d
from .map_1d_to_3d import map_1d_to_3d

__all__ = ['map_3d_to_1d', 'map_1d_to_3d']