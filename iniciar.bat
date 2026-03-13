@echo off
:: ─────────────────────────────────────────────────────────────────
::  Launcher do NêveBot
::  Sempre usa o Python do venv e garante instância única.
:: ─────────────────────────────────────────────────────────────────
cd /d "%~dp0"

:: Mata qualquer processo Python rodando nevebot.py
taskkill /F /FI "WINDOWTITLE eq nevebot*" >nul 2>&1
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO CSV ^| findstr /i "python"') do (
    wmic process where "ProcessId=%%~a" get CommandLine 2>nul | findstr /i "nevebot.py" >nul && taskkill /F /PID %%~a >nul 2>&1
)
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python3.exe" /FO CSV ^| findstr /i "python"') do (
    wmic process where "ProcessId=%%~a" get CommandLine 2>nul | findstr /i "nevebot.py" >nul && taskkill /F /PID %%~a >nul 2>&1
)

:: Aguarda portas liberarem
timeout /t 2 /nobreak >nul

echo Iniciando NêveBot... (Ctrl+C para desligar)
echo.
venv\Scripts\python.exe -u nevebot.py
