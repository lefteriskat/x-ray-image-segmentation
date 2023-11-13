#!/bin/bash

# Delete the virtual environment if exists
rm -rf venv

# Create a virtual environment
python3.11 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Install requirements
pip3 install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Environment setup complete."
