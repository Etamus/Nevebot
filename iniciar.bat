@echo off
:: ─────────────────────────────────────────────────────────────────
::  Launcher do Nevebot
::  Sempre usa o Python do venv e garante instancia unica.
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

:: Variáveis para melhor compatibilidade CUDA e relatório de erros
set PYTHONFAULTHANDLER=1
set GGML_CUDA_NO_PINNED=1
set PYTHONUTF8=1

mkdir logs >nul 2>&1

echo Iniciando Nevebot... (Ctrl+C para desligar)
echo.
venv\Scripts\python.exe -u nevebot.py

echo.
echo Bot encerrado. Erros salvos em logs\nevebot_error.log
pause
