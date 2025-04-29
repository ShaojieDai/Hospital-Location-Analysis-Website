@echo off
echo Starting Hospital Location Analyzer...

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    pause
    exit /b
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please install the venv module and try again.
        pause
        exit /b
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check if OpenAI API key is set
findstr /C:"your_openai_api_key_here" .env >nul
if %errorlevel% equ 0 (
    echo.
    echo WARNING: The OpenAI API key has not been set.
    echo To enable AI analysis, please edit the .env file and replace 'your_openai_api_key_here' with your actual OpenAI API key.
    echo.
    set /p answer=Continue without API key? (y/n):
    if /i not "%answer%"=="y" (
        echo Exiting. Please set the API key in the .env file and try again.
        pause
        exit /b
    )
)

REM Run the application
echo Starting Flask application...
python app.py

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause
