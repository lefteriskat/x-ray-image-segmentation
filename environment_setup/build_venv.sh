#!/bin/bash

# Prompt user for Python version installation
read -p "Do you want to install Python 3.11.0? (y/n): " install_python

if [ "$install_python" = "y" ]; then
    # Define the desired Python version
    PYTHON_VERSION="3.11.0"

    # Uninstall existing Python version
    if command -v pyenv &> /dev/null; then
        # Check if Python version is already installed
        if pyenv versions --bare | grep -q "$PYTHON_VERSION"; then
            echo "Uninstalling existing Python $PYTHON_VERSION..."
            pyenv uninstall "$PYTHON_VERSION"
        fi
    fi

    # Install Python version using pyenv
    curl https://pyenv.run | zsh
    export PATH="$HOME/.pyenv/bin:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"

    pyenv install "$PYTHON_VERSION"
    pyenv global "$PYTHON_VERSION"
fi

# Delete the virtual environment if it exists
rm -rf venv

# Install additional dependencies
sudo apt-get install "python3.11-venv" "python3.11-dev"

# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate
echo "Environment created and activated."

# Install requirements
pip3 install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Environment setup complete."
