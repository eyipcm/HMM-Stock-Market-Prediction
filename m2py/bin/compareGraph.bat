@echo off
REM HMM Stock Market Prediction - Graph Comparison Batch File
REM This batch file runs the comparison script with absolute paths

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
