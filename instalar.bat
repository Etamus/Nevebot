@echo off
setlocal EnableExtensions

cd /d "%~dp0"

echo ================================================
echo  Nevebot - instalador
echo ================================================
echo.

if /I "%~1"=="--llama-only" goto LLAMA_CPP

if not exist "venv\Scripts\python.exe" (
    echo Criando ambiente virtual em venv...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.11 -m venv venv
        if errorlevel 1 py -3 -m venv venv
    ) else (
        python -m venv venv
    )
)

if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Nao foi possivel criar/encontrar o ambiente virtual.
    echo Instale Python 3.11+ e tente novamente.
    pause
    exit /b 1
)

echo Ativando ambiente virtual...
call "venv\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERRO] Falha ao ativar o ambiente virtual.
    pause
    exit /b 1
)

echo Atualizando pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [ERRO] Falha ao atualizar ferramentas do pip.
    pause
    exit /b 1
)

echo Removendo llama-cpp-python antigo, se existir...
python -m pip uninstall -y llama-cpp-python llama_cpp_python >nul 2>&1

echo Instalando dependencias do requirements.txt...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo Garantindo dependencias de voz...
python -m pip install omnivoice faster-whisper ctranslate2 onnxruntime-gpu
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias de voz.
    pause
    exit /b 1
)

:LLAMA_CPP
echo.
echo Baixando/atualizando llama.cpp oficial do GitHub...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\baixar_llama_cpp.ps1"
if errorlevel 1 (
    echo [ERRO] Falha ao baixar/instalar llama.cpp.
    if /I "%~1"=="--llama-only" exit /b 1
    pause
    exit /b 1
)

if /I "%~1"=="--llama-only" exit /b 0

echo.
echo Baixando modelos Kokoro auxiliares...
if not exist "models\kokoro" mkdir "models\kokoro"
if not exist "models\kokoro\kokoro-v1.0.fp16.onnx" (
    echo Baixando kokoro-v1.0.fp16.onnx...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx' -OutFile 'models\kokoro\kokoro-v1.0.fp16.onnx'"
    if errorlevel 1 (
        echo [ERRO] Falha ao baixar kokoro-v1.0.fp16.onnx.
        pause
        exit /b 1
    )
)
if not exist "models\kokoro\voices-v1.0.bin" (
    echo Baixando voices-v1.0.bin...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' -OutFile 'models\kokoro\voices-v1.0.bin'"
    if errorlevel 1 (
        echo [ERRO] Falha ao baixar voices-v1.0.bin.
        pause
        exit /b 1
    )
)

echo.
echo Instalacao concluida com sucesso.
echo Use iniciar.bat para iniciar o bot.
pause
exit /b 0
