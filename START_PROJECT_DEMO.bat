@echo off
TITLE MultiAgent GrapeRAG - Project Showcase Launcher
COLOR 0A
cls
echo =====================================================================
echo           MULTI-AGENT ADAPTIVE GRAPHRAG - PROJECT SHOWCASE
echo =====================================================================
echo.
echo Choose an option to showcase to your teachers / guide:
echo.
echo  [1] Launch Streamlit Visual Dashboard (RECOMMENDED for presentation)
echo      - Displays live Knowledge Graph visualization (GraphViz)
echo      - Shows dynamic routing complexity C(Q) badges
echo      - Demonstrates Multi-Hop PDF Reasoning & Verification Scorecard
echo.
echo  [2] Launch Full-Stack Web App (Next.js + FastAPI)
echo      - Starts FastAPI Backend on http://localhost:8000
echo      - Starts Next.js Modern Frontend on http://localhost:3000
echo.
echo  [3] Run Multi-Hop Terminal Demo (Instant Proof of Concept)
echo      - Traverses Doc 1 -> Doc 2 -> Doc 3 and synthesizes answer
echo.
echo  [4] Open Final Formatted Project Report (project_final_formatted_v4.docx)
echo.
echo  [5] Exit
echo.
echo =====================================================================
set /p choice="Enter your choice (1, 2, 3, 4, or 5): "

if "%choice%"=="1" goto STREAMLIT
if "%choice%"=="2" goto NEXTJS
if "%choice%"=="3" goto TEST
if "%choice%"=="4" goto REPORT
if "%choice%"=="5" goto EXIT

:STREAMLIT
echo.
echo Starting Streamlit Visual Dashboard on http://localhost:8501 ...
cd /d "C:\Users\KETAN\Downloads\aryan project\MultiAgent_GrapeRAG\frontend\api"
start http://localhost:8501
.\venv\Scripts\streamlit.exe run ui\app.py
pause
goto EXIT

:NEXTJS
echo.
echo Starting FastAPI Backend on http://localhost:8000 ...
cd /d "C:\Users\KETAN\Downloads\aryan project\MultiAgent_GrapeRAG\frontend\api"
start "FastAPI Backend" cmd /k ".\venv\Scripts\python.exe -m uvicorn index:app --reload --port 8000"

timeout /t 2 >nul

echo Starting Next.js Frontend on http://localhost:3000 ...
cd /d "C:\Users\KETAN\Downloads\aryan project\MultiAgent_GrapeRAG\frontend"
start http://localhost:3000
npm run dev
pause
goto EXIT

:TEST
echo.
echo Running Multi-Hop Reasoning CLI Test across 3 sample PDFs...
cd /d "C:\Users\KETAN\Downloads\aryan project\MultiAgent_GrapeRAG\frontend\api"
.\venv\Scripts\python.exe test_query.py
echo.
pause
goto EXIT

:REPORT
echo.
echo Opening Final Project Report in Microsoft Word...
start "" "C:\Users\KETAN\OneDrive\Desktop\project_final_formatted_v4.docx"
pause
goto EXIT

:EXIT
exit
