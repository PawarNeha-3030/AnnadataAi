@echo off
echo Installing FarmAI Dependencies...
echo.

REM Check Python
python --version
if errorlevel 1 (
    echo Python not found! Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo Installing Flask and web dependencies...
python -m pip install --upgrade pip
python -m pip install Flask==2.3.3
python -m pip install Flask-CORS==4.0.0
python -m pip install Pillow==10.0.0

echo Installing ML dependencies...
python -m pip install numpy==1.24.3
python -m pip install opencv-python==4.8.1.78
python -m pip install tensorflow==2.13.0

echo.
echo ✅ All dependencies installed successfully!
echo.
echo To run the application:
echo 1. Save all HTML files in templates/ folder
echo 2. Run: python app.py
echo 3. Open: http://localhost:5000/diseases
echo.
pause