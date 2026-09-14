@echo off
REM ===========================================================================
REM  Smart Municipal Waste Segregation -- local launcher (Windows)
REM
REM  Double-click this file. It builds an isolated environment the first time
REM  (a few minutes, mostly torch), then offers the three ways to run.
REM
REM  It installs requirements-LOCAL.txt, not requirements.txt: the deployment
REM  list uses headless OpenCV, which cannot open main.py's window.
REM ===========================================================================
setlocal
cd /d "%~dp0"

echo.
echo  Smart Municipal Waste Segregation -- local launcher
echo  ==================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo  [X] Python was not found on your PATH.
    echo      Install Python 3.11 or newer from python.org and tick
    echo      "Add python.exe to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo  [1/2] Creating an isolated environment in .venv ...
    python -m venv .venv
    if errorlevel 1 goto :venvfail
    echo  [2/2] Installing dependencies. First run downloads torch, so this
    echo        takes a few minutes. Later runs skip straight past it.
    echo.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements-local.txt
    if errorlevel 1 goto :pipfail
    echo.
    echo  Environment ready.
    echo.
)

:menu
echo.
echo  What would you like to run?
echo.
echo    1  Live desktop loop        - your webcam, the moving belt, full speed
echo    2  Web app                  - the three-page Streamlit site, in a browser
echo    3  Seed demo data           - fill the database so the report has history
echo    4  Run the test suite       - all 253 checks, no camera needed
echo    5  Quit
echo.
set /p choice="  Choose 1-5: "

if "%choice%"=="1" goto :live
if "%choice%"=="2" goto :web
if "%choice%"=="3" goto :seed
if "%choice%"=="4" goto :tests
if "%choice%"=="5" exit /b 0
echo  Please choose a number from 1 to 5.
goto :menu

:live
echo.
echo  Starting the live loop. Point the webcam at a bottle, banana, book,
echo  phone, cup or pair of scissors.
echo.
echo    Q  quit        S  start/stop the belt
echo    E  empty bins  F  inject actuator failures
echo.
".venv\Scripts\python.exe" main.py
goto :done

:web
echo.
echo  Starting the web app. Your browser should open automatically;
echo  if not, go to http://localhost:8501
echo.
echo  Press Ctrl+C in this window to stop it.
echo.
".venv\Scripts\python.exe" -m streamlit run streamlit_app.py
goto :done

:seed
echo.
".venv\Scripts\python.exe" -m tests.seed_demo_data --items 60 --reset
echo.
echo  Done. Open the web app and look at the reporting view.
goto :done

:tests
echo.
for %%T in (test_pipeline test_database test_analytics test_labels test_hardware test_evaluation test_theme) do (
    echo  --- %%T
    ".venv\Scripts\python.exe" -m tests.%%T
)
goto :done

:venvfail
echo.
echo  [X] Could not create the virtual environment.
echo      Check that your Python install is not the Microsoft Store stub.
echo.
pause
exit /b 1

:pipfail
echo.
echo  [X] Dependency install failed. The usual causes are no internet
echo      connection or not enough disk space (torch needs ~1 GB).
echo.
pause
exit /b 1

:done
echo.
pause
goto :menu
