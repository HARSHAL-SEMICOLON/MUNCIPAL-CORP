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
echo.
echo    --- getting your results chapter ---
echo    3  Capture evaluation frames - camera opens, SPACE saves, Q done
echo    4  Show what you have captured
echo    5  Run the evaluation        - accuracy, wrong-bin rate, confusion matrix
echo    6  Choose demo samples       - picks the frames the detector sees best
echo.
echo    7  Seed demo data           - fill the database so the report has history
echo    8  Run the test suite       - all 253 checks, no camera needed
echo    9  Quit
echo.
set /p choice="  Choose 1-9: "

if "%choice%"=="1" goto :live
if "%choice%"=="2" goto :web
if "%choice%"=="3" goto :capture
if "%choice%"=="4" goto :inventory
if "%choice%"=="5" goto :evaluate
if "%choice%"=="6" goto :samples
if "%choice%"=="7" goto :seed
if "%choice%"=="8" goto :tests
if "%choice%"=="9" exit /b 0
echo  Please choose a number from 1 to 9.
goto :menu

:capture
echo.
echo  Which class are you about to photograph?
echo.
echo  Good ones for the current COCO model: bottle, banana, apple, orange,
echo  book, cup, bowl, cell phone, laptop, scissors, wine glass.
echo.
echo  Battery, aluminium can, cardboard and plastic bag are NOT classes this
echo  model knows. Capture them anyway if you like - a miss is a real result
echo  and the evaluation counts it - but expect no box to appear.
echo.
set /p lbl="  Class name (e.g. bottle): "
if "%lbl%"=="" goto :menu
echo.
echo  SPACE saves a frame.  Q finishes.
echo  Aim for 20+ frames: vary the angle, the distance and the lighting.
echo.
".venv\Scripts\python.exe" -m evaluation.capture --label "%lbl%"
goto :done

:inventory
echo.
".venv\Scripts\python.exe" -m evaluation.capture --label x --list
goto :done

:evaluate
echo.
echo  Running the real detector and the real agents over every captured frame.
echo.
".venv\Scripts\python.exe" -m evaluation.evaluate
echo.
echo  Results written to evaluation\results\evaluation.json
echo  Put the headline numbers in your README - that table is the thing the
echo  project has been missing.
goto :done

:samples
echo.
".venv\Scripts\python.exe" -m tools.make_samples
goto :done

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
