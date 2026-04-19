@echo off
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Erro ao ativar o ambiente virtual. Verifique se o venv existe.
    pause
    exit /b 1
)

echo Instalando llama-cpp-python v0.3.4 com suporte CUDA 12.4...
pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp311-cp311-win_amd64.whl"
if %errorlevel% neq 0 (
    echo Erro ao instalar llama-cpp-python.
    pause
    exit /b 1
)

echo Instalando dependencias do requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Erro ao instalar dependencias.
    pause
    exit /b 1
)

echo Instalando OmniVoice (TTS)...
pip install omnivoice
if %errorlevel% neq 0 (
    echo Erro ao instalar OmniVoice.
    pause
    exit /b 1
)

echo Baixando modelos Kokoro...
if not exist models\kokoro mkdir models\kokoro
if not exist models\kokoro\kokoro-v1.0.fp16.onnx (
    echo Baixando kokoro-v1.0.fp16.onnx...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.fp16.onnx' -OutFile 'models\kokoro\kokoro-v1.0.fp16.onnx'"
    if %errorlevel% neq 0 (
        echo Erro ao baixar kokoro-v1.0.fp16.onnx.
        pause
        exit /b 1
    )
)
if not exist models\kokoro\voices-v1.0.bin (
    echo Baixando voices-v1.0.bin...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin' -OutFile 'models\kokoro\voices-v1.0.bin'"
    if %errorlevel% neq 0 (
        echo Erro ao baixar voices-v1.0.bin.
        pause
        exit /b 1
    )
)

echo Instalacao concluida com sucesso!
pause