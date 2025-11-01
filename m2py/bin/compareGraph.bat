@echo off
:: HMM Stock Market Prediction - Graph Comparison Batch File
:: This batch file runs the comparison script with absolute paths

:: SPDX-FileCopyrightText: Copyright (C) 2025 Ernest YIP <eyipcm@gmail.com>
:: SPDX-License-Identifier: GPL-3.0-or-later

:: This program is free software: you can redistribute it and/or modify
:: it under the terms of the GNU General Public License as published by
::the Free Software Foundation, either version 3 of the License, or
::(at your option) any later version.

:: This program is distributed in the hope that it will be useful,
:: but WITHOUT ANY WARRANTY; without even the implied warranty of
:: MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
:: GNU General Public License for more details.

:: You should have received a copy of the GNU General Public License
:: along with this program.  If not, see <https://www.gnu.org/licenses/>.

echo ============================================================
echo HMM Stock Market Prediction - Python vs MATLAB Comparison
echo ============================================================
echo.

REM Set the working directory to the utils folder
cd /d "D:\gitrepo\HMM-Stock-Market-Prediction\m2py\utils"

REM Run the comparison script with absolute paths
python comparison_main.py ^
    --python-output "D:\gitrepo\HMM-Stock-Market-Prediction\m2py\output_figs" ^
    --matlab-output "D:\gitrepo\HMM-Stock-Market-Prediction\out_figs_png" ^
    --output-dir "D:\gitrepo\HMM-Stock-Market-Prediction\m2py\utils\comparison_results" ^
    --verbose

echo.
echo ============================================================
echo Comparison completed!
echo Results saved to: D:\gitrepo\HMM-Stock-Market-Prediction\m2py\utils\comparison_results
echo Report: D:\gitrepo\HMM-Stock-Market-Prediction\m2py\utils\comparison_results\comparison_report.md
echo ============================================================
echo.
pause
