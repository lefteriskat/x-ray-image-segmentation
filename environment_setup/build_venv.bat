#!/bin/bash

# Prompt user for Python version installation
read -p "Do you want to install Python 3.11.0? (y/n): " install_python

# Define the desired Python version
PYTHON_VERSION="3.11.0"

if [ "$install_python" = "y" ]; then
    # Uninstall existing Python version
    if command -v pyenv &> /dev/null; then
        # Check if Python version is already installed
        if pyenv versions --bare | grep -q "$PYTHON_VERSION"; then
            echo "Uninstalling existing Python $PYTHON_VERSION..."
            pyenv uninstall "$PYTHON_VERSION"
        fi
    fi

    # Install pyenv
    curl https://pyenv.run | bash
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"

    # Install Python 3.11.0
    pyenv install "$PYTHON_VERSION"
    pyenv global "$PYTHON_VERSION"
else
    # Use existing Python version
fi

# Remove existing virtual environment
rm -rf venv

# Get virtualenv
python -m pip install virtualenv==20.17.0
if [ $? -ne 0 ]; then
    # Handle virtualenv installation error
    echo "An error occurred during the virtualenv installation. Running error handling code..."
    # removing the certifications
    python -m pip install virtualenv==20.17.0 --trusted-host pypi.org --trusted-host files.pythonhosted.org
    echo "virtualenv installation error handling code executed."
    exit 1
fi

# Create virtual environment
virtualenv venv -p python3.11

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    # Handle requirements installation error
    echo "An error occurred during the requirements installation. Running error handling code..."
    # removing the certifications
    pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
    echo "requirements installation error handling code executed."
    exit 2
fi

# If all steps succeeded, exit the script
exit 0


:HandleError1
echo An error occurred during the virtualenv installation. Running error handling code...
REM removing the certifications
python -m pip install virtualenv==20.17.0 --trusted-host pypi.org --trusted-host files.pythonhosted.org
echo virtualenv installation error handling code executed.
pause

:HandleError2
echo An error occurred during the requirements installation. Running error handling code...
REM removing the certifications
pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
echo requirements installation error handling code executed.
pause
