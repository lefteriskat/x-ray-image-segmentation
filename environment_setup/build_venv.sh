#!/bin/bash

# Prompt user for Python version installation
read -p "Do you want to install Python 3.12.0? (y/n): " install_python

if [ "$install_python" = "y" ]; then
    # Define the desired Python version
    PYTHON_VERSION="3.12.0"

    # Install Python version using pyenv
    curl https://pyenv.run | zsh
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"

    pyenv install $PYTHON_VERSION
    pyenv global $PYTHON_VERSION
fi

# Delete the virtual environment if it exists
rm -rf venv

# Install additional dependencies
sudo apt-get install python3.12-venv python3.12-dev

# Create a virtual environment
python3.12 -m venv venv

# Activate the virtual environment
. venv/bin/activate
echo "Environment created and activated."

# Install requirements
pip3 install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Environment setup complete."
