@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "CUDA_PATH="

echo ================================================
echo  Nevebot - instalador
echo ================================================
echo.

if /I "%~1"=="--llama-only" goto LLAMA_CPP

if not exist "requirements.txt" (
    echo [ERRO] requirements.txt nao encontrado.
    pause
    exit /b 1
)

echo Preparando pastas locais...
for %%D in ("data" "gravacoes" "logs" "models" "models\texto" "models\chatterbox") do (
    if not exist "%%~D" mkdir "%%~D"
)

if not exist ".env" (
    if exist ".env.example" (
        copy /Y ".env.example" ".env" >nul
        echo .env criado a partir de .env.example.
    ) else (
        echo [AVISO] .env.example nao encontrado; crie um .env com DISCORD_TOKEN.
    )
)

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

set "PY=%CD%\venv\Scripts\python.exe"
echo Usando Python do venv: %PY%

"%PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo [ERRO] Python 3.11+ e necessario dentro do venv.
    echo Versao encontrada:
    "%PY%" --version
    pause
    exit /b 1
)

echo Verificando Microsoft Edge WebView2 Runtime...
powershell -NoProfile -Command "$p=@('HKCU:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'); if ($p.Where({Test-Path $_}).Count) { exit 0 }; exit 1" >nul 2>&1
if errorlevel 1 (
    where winget >nul 2>&1
    if not errorlevel 1 (
        echo Instalando WebView2 Runtime para a interface nativa...
        winget install --id Microsoft.EdgeWebView2Runtime -e --silent --accept-package-agreements --accept-source-agreements
    )
    powershell -NoProfile -Command "$p=@('HKCU:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'); if ($p.Where({Test-Path $_}).Count) { exit 0 }; exit 1" >nul 2>&1
    if errorlevel 1 echo [AVISO] WebView2 indisponivel; a interface usara o navegador padrao.
) else (
    echo WebView2 Runtime encontrado.
)

echo Atualizando pip/setuptools/wheel...
"%PY%" -m pip install --upgrade pip wheel "setuptools<81"
if errorlevel 1 (
    echo [ERRO] Falha ao atualizar ferramentas do pip.
    pause
    exit /b 1
)

echo Removendo llama-cpp-python antigo, se existir...
"%PY%" -m pip uninstall -y llama-cpp-python llama_cpp_python >nul 2>&1

echo Removendo Chatterbox antigo antes de resolver dependencias...
"%PY%" -m pip uninstall -y chatterbox-tts >nul 2>&1

echo Preparando PyTorch para GPU, se disponivel...
"%PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    if exist "%WINDIR%\System32\nvcuda.dll" (
        echo NVIDIA detectada; instalando PyTorch CUDA cu128...
        "%PY%" -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128 torchaudio==2.10.0+cu128
        if errorlevel 1 (
            echo [AVISO] Falha ao instalar PyTorch CUDA. Tentando PyTorch CPU.
            "%PY%" -m pip install torch==2.10.0 torchaudio==2.10.0
        )
    ) else (
        echo [AVISO] NVIDIA nao detectada. Instalando PyTorch CPU.
        "%PY%" -m pip install torch==2.10.0 torchaudio==2.10.0
    )
    if errorlevel 1 (
        echo [ERRO] Falha ao instalar PyTorch/torchaudio.
        pause
        exit /b 1
    )
)

echo Instalando dependencias do requirements.txt...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo Instalando pacote Chatterbox PT-BR...
"%PY%" -m pip install chatterbox-tts==0.1.7 --no-deps
if errorlevel 1 (
    echo [ERRO] Falha ao instalar chatterbox-tts.
    pause
    exit /b 1
)

echo Validando dependencias principais...
"%PY%" -c "import discord, faster_whisper, chatterbox, torch, webview; print('Dependencias principais OK.'); print('Torch CUDA:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERRO] Alguma dependencia principal nao importou corretamente.
    pause
    exit /b 1
)

"%PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] PyTorch nao detectou CUDA. STT/TTS podem funcionar na CPU, mas ficarao mais lentos.
)

:LLAMA_CPP
echo.
if not exist "scripts\baixar_llama_cpp.ps1" (
    echo [ERRO] scripts\baixar_llama_cpp.ps1 nao encontrado.
    if /I "%~1"=="--llama-only" exit /b 1
    pause
    exit /b 1
)

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
if not exist "scripts\preparar_chatterbox_ptbr.py" (
    echo [ERRO] scripts\preparar_chatterbox_ptbr.py nao encontrado.
    pause
    exit /b 1
)

echo Baixando modelos Chatterbox PT-BR...
"%PY%" scripts\preparar_chatterbox_ptbr.py
if errorlevel 1 (
    echo [ERRO] Falha ao baixar modelos Chatterbox PT-BR.
    pause
    exit /b 1
)

dir /b "models\texto\*.gguf" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [AVISO] Nenhum modelo .gguf encontrado em models\texto.
    echo Coloque seu modelo LLM GGUF nessa pasta ou ajuste LLM_MODEL_PATH no .env.
)

if exist ".env" (
    findstr /B /C:"DISCORD_TOKEN=SEU_TOKEN_AQUI" ".env" >nul 2>&1
    if not errorlevel 1 (
        echo.
        echo [AVISO] Edite o arquivo .env e troque DISCORD_TOKEN=SEU_TOKEN_AQUI pelo token real.
    )
)

echo.
echo Instalacao concluida com sucesso.
echo Use iniciar.bat para iniciar o bot.
pause
exit /b 0
