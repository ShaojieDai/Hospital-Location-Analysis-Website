#!/bin/bash

echo "Starting Hospital Location Analyzer..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install the venv module and try again."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/Mac
    source venv/bin/activate
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if OpenAI API key is set
if grep -q "your_openai_api_key_here" .env; then
    echo ""
    echo "WARNING: The OpenAI API key has not been set."
    echo "To enable AI analysis, please edit the .env file and replace 'your_openai_api_key_here' with your actual OpenAI API key."
    echo ""
    read -p "Continue without API key? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please set the API key in the .env file and try again."
        exit 1
    fi
fi

# Run the application
echo "Starting Flask application..."
python app.py

# Deactivate virtual environment
deactivate
