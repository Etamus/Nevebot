@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ================================================
echo  Nevebot
echo ================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Python do ambiente virtual nao encontrado.
    echo Execute instalar.bat ou recrie o venv antes de iniciar.
    echo.
    pause
    exit /b 1
)

set "PY=%CD%\venv\Scripts\python.exe"
mkdir logs >nul 2>&1

if not exist "scripts\validar_runtime.py" (
    echo [ERRO] Validador do runtime nao encontrado.
    echo Execute instalar.bat para completar os arquivos do projeto.
    echo.
    pause
    exit /b 1
)

"%PY%" "scripts\validar_runtime.py" >"logs\runtime-check.log" 2>&1
if errorlevel 1 (
    echo O ambiente Python esta incompleto. Iniciando reparo automatico...
    type "logs\runtime-check.log"
    echo.
    call "%~dp0instalar.bat" --repair-runtime
    if errorlevel 1 (
        echo [ERRO] O reparo automatico do ambiente falhou.
        echo Confira logs\runtime-check.log e execute instalar.bat se necessario.
        echo.
        pause
        exit /b 1
    )
    "%PY%" "scripts\validar_runtime.py" >"logs\runtime-check.log" 2>&1
    if errorlevel 1 (
        echo [ERRO] O runtime ainda esta incompleto apos o reparo.
        type "logs\runtime-check.log"
        echo.
        pause
        exit /b 1
    )
)

if not exist "llama.cpp\llama-server.exe" (
    echo llama.cpp local nao encontrado. Baixando a ultima release oficial...
    call "%~dp0instalar.bat" --llama-only
    if errorlevel 1 (
        echo [ERRO] Nao foi possivel preparar o llama.cpp local.
        echo Execute instalar.bat e tente novamente.
        echo.
        pause
        exit /b 1
    )
)

if exist "logs\ui_shutdown.flag" del /q "logs\ui_shutdown.flag" >nul 2>&1

echo Encerrando instancias antigas do Nevebot...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-CimInstance Win32_Process -Filter \"name = 'python.exe' or name = 'python3.exe'\" | Where-Object { $_.CommandLine -match 'nevebot\.py' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1
powershell -NoProfile -ExecutionPolicy Bypass -Command "$target=[IO.Path]::GetFullPath('%CD%\llama.cpp\llama-server.exe'); Get-CimInstance Win32_Process -Filter \"name = 'llama-server.exe'\" | Where-Object { $_.ExecutablePath -and [IO.Path]::GetFullPath($_.ExecutablePath) -eq $target } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

set "PYTHONFAULTHANDLER=1"
set "GGML_CUDA_NO_PINNED=1"
set "PYTHONUTF8=1"
set "LLAMA_CPP_DIR=%CD%\llama.cpp"
set "LLAMA_CPP_SERVER_EXE=%CD%\llama.cpp\llama-server.exe"
set "CUDA_PATH="
set "NEVEBOT_PREWARM_VOICE=1"

echo Iniciando Nevebot... use Ctrl+C para desligar.
echo.
"%PY%" -u nevebot.py
set "EXIT_CODE=%ERRORLEVEL%"

if exist "logs\ui_shutdown.flag" (
    del /q "logs\ui_shutdown.flag" >nul 2>&1
    exit /b 0
)

echo.
echo Bot encerrado com codigo %EXIT_CODE%.
echo Se houve erro, confira logs\nevebot_error.log.
echo.
pause
exit /b %EXIT_CODE%
