@echo off
echo ==================================================
echo           CROP YIELD & FERTILIZER RUNNER
echo ==================================================
echo.
echo Please choose an option:
echo   [1] Run Crop Yield Prediction (train_pipeline.py)
echo   [2] Run Fertilizer Recommendation (train_fertilizer.py)
echo   [3] Install Requirements (requirements.txt)
echo   [4] Exit
echo.
set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo.
    echo Running train_pipeline.py...
    python train_pipeline.py
    pause
    goto end
)
if "%choice%"=="2" (
    echo.
    echo Running train_fertilizer.py...
    python train_fertilizer.py
    pause
    goto end
)
if "%choice%"=="3" (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    pause
    goto end
)
if "%choice%"=="4" (
    goto end
)

:end
echo.
echo Goodbye!
