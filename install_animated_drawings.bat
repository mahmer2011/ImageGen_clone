@echo off
REM Install AnimatedDrawings with Visual Studio 2026 Insiders build environment
echo Setting up Visual Studio 2026 Insiders build environment...
call "C:\Program Files\Microsoft Visual Studio\18\Insiders\VC\Auxiliary\Build\vcvars64.bat"

echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing animated_drawings (this may take 5-10 minutes)...
pip install git+https://github.com/facebookresearch/AnimatedDrawings.git

echo.
echo Installation complete!
pause

