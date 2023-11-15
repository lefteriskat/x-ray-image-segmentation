@echo off

REM Prompt user for Python version installation
set /p install_python=Do you want to install Python 3.12.0? (y/n): 

IF /I "%install_python%" EQU "y" (
    REM Install pyenv
    curl https://pyenv.run | cmd
    set PATH=%USERPROFILE%\.pyenv\bin;%PATH%
    pyenv init --path | cmd
    pyenv virtualenv-init - | cmd

    REM Install Python 3.12.0
    pyenv install 3.12.0
    pyenv global 3.12.0
) ELSE (
    REM Use existing Python version
)

REM Remove existing virtual environment
RMDIR /Q /S venv 2>nul

REM Get virtualenv
python -m pip install virtualenv==20.17.0
IF %ERRORLEVEL% NEQ 0 GOTO HandleError1

REM Create virtual environment
virtualenv venv -p python3.12

REM Activate virtual environment
CALL venv\Scripts\activate

REM Install requirements
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 GOTO HandleError2

REM If all steps succeeded, exit the script
exit /b

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
