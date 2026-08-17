@echo off
setlocal EnableExtensions

cd /d "%~dp0"
set "CUDA_PATH="
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PIP_DEFAULT_TIMEOUT=90"
set "TORCH_VERSION=2.11.0"

echo ================================================
echo  Nevebot - instalador
echo ================================================
echo.

if /I "%~1"=="--llama-only" goto LLAMA_CPP_ONLY
if /I "%~1"=="--check" goto CHECK_ONLY

for %%F in (
    "requirements.txt"
    ".env.example"
    "scripts\baixar_llama_cpp.ps1"
    "scripts\preparar_chatterbox_ptbr.py"
    "scripts\preparar_whisper.py"
    "scripts\validar_instalacao.py"
) do (
    if not exist "%%~F" (
        echo [ERRO] Arquivo obrigatorio ausente: %%~F
        goto FALHA
    )
)

call :VERIFICAR_ESPACO
if errorlevel 1 goto FALHA

echo Preparando pastas locais...
for %%D in (
    "data"
    "gravacoes"
    "transcricoes"
    "logs"
    "models"
    "models\texto"
    "models\whisper"
    "models\chatterbox"
) do (
    if not exist "%%~D" mkdir "%%~D"
    if not exist "%%~D" (
        echo [ERRO] Nao foi possivel criar a pasta %%~D.
        goto FALHA
    )
)

if not exist ".env" (
    copy /Y ".env.example" ".env" >nul
    if errorlevel 1 (
        echo [ERRO] Nao foi possivel criar .env a partir de .env.example.
        goto FALHA
    )
    echo .env criado a partir de .env.example.
) else (
    echo .env existente preservado.
)

call :LOCALIZAR_PYTHON
if errorlevel 1 goto FALHA

call :PREPARAR_VENV
if errorlevel 1 goto FALHA

set "PY=%CD%\venv\Scripts\python.exe"
echo Usando Python do venv: %PY%

call :PREPARAR_WEBVIEW2

echo Atualizando pip, setuptools e wheel...
"%PY%" -m pip install --upgrade --retries 5 --timeout 90 pip wheel "setuptools<81"
if errorlevel 1 (
    echo [ERRO] Falha ao atualizar as ferramentas do pip.
    goto FALHA
)

echo Removendo pacotes antigos que conflitam com o runtime atual...
"%PY%" -m pip uninstall -y llama-cpp-python llama_cpp_python chatterbox-tts >nul 2>&1

call :PREPARAR_PYTORCH
if errorlevel 1 goto FALHA

echo Instalando dependencias do requirements.txt...
"%PY%" -m pip install --retries 5 --timeout 90 -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar as dependencias do projeto.
    goto FALHA
)

echo Instalando o pacote Chatterbox PT-BR...
"%PY%" -m pip install --retries 5 --timeout 90 chatterbox-tts==0.1.7 --no-deps
if errorlevel 1 (
    echo [ERRO] Falha ao instalar chatterbox-tts.
    goto FALHA
)

echo.
echo Baixando o modelo de reconhecimento de voz...
"%PY%" "scripts\preparar_whisper.py"
if errorlevel 1 (
    echo [ERRO] Falha ao preparar o modelo do Whisper.
    goto FALHA
)

echo.
echo Baixando os modelos Chatterbox PT-BR...
"%PY%" "scripts\preparar_chatterbox_ptbr.py"
if errorlevel 1 (
    echo [ERRO] Falha ao baixar os modelos Chatterbox PT-BR.
    goto FALHA
)

if not defined LLAMA_CPP_BACKEND (
    "%PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        set "LLAMA_CPP_BACKEND=cpu"
    ) else (
        set "LLAMA_CPP_BACKEND=cuda"
    )
)

call :INSTALAR_LLAMA
if errorlevel 1 goto FALHA

echo.
echo Validando a instalacao completa...
"%PY%" "scripts\validar_instalacao.py"
if errorlevel 1 (
    echo [ERRO] A validacao tecnica encontrou problemas.
    goto FALHA
)

echo.
echo ================================================
echo  Instalacao tecnica concluida
echo ================================================
echo Os itens marcados como PENDENTE pelo validador precisam ser fornecidos
echo pelo usuario para o uso completo: GGUF e voz de referencia.
echo O token do Discord pode ser informado depois pela interface do Nevebot.
echo Quando nao houver pendencias, use iniciar.bat.
goto SUCESSO


:VERIFICAR_ESPACO
echo Verificando espaco livre em disco...
powershell -NoProfile -Command "$drive=[IO.Path]::GetPathRoot((Get-Location).Path).Substring(0,1); $free=(Get-PSDrive -Name $drive).Free; Write-Host ('Espaco livre: {0:N1} GB' -f ($free/1GB)); if($free -lt 10GB){exit 2}; if($free -lt 18GB){exit 1}; exit 0"
if errorlevel 2 (
    echo [ERRO] Menos de 10 GB livres. Libere espaco antes de instalar.
    exit /b 1
)
if errorlevel 1 echo [AVISO] Menos de 18 GB livres; um modelo GGUF grande pode nao caber.
exit /b 0


:LOCALIZAR_PYTHON
set "PY_BOOT="
set "PY_BOOT_ARGS="

py -3.11 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_BOOT=py"
    set "PY_BOOT_ARGS=-3.11"
    exit /b 0
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PY_BOOT=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        exit /b 0
    )
)

python -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_BOOT=python"
    exit /b 0
)

py -3 -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_BOOT=py"
    set "PY_BOOT_ARGS=-3"
    exit /b 0
)

where winget >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python 3.11 de 64 bits nao foi encontrado e o winget nao esta disponivel.
    echo Instale Python 3.11 em https://www.python.org/downloads/ e execute novamente.
    exit /b 1
)

echo Python compativel nao encontrado. Instalando Python 3.11 para o usuario atual...
winget install --id Python.Python.3.11 -e --scope user --silent --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo [ERRO] O winget nao conseguiu instalar Python 3.11.
    exit /b 1
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PY_BOOT=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    exit /b 0
)

py -3.11 -c "import struct; raise SystemExit(0 if struct.calcsize('P')*8==64 else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PY_BOOT=py"
    set "PY_BOOT_ARGS=-3.11"
    exit /b 0
)

echo [ERRO] Python foi instalado, mas nao ficou acessivel nesta sessao.
echo Feche este console, abra instalar.bat novamente e tente de novo.
exit /b 1


:PREPARAR_VENV
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
    if errorlevel 1 (
        echo Ambiente virtual incompatível encontrado. Recriando venv...
        rmdir /S /Q "venv"
    )
)

if not exist "venv\Scripts\python.exe" (
    echo Criando ambiente virtual em venv...
    "%PY_BOOT%" %PY_BOOT_ARGS% -m venv "venv"
)

if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Nao foi possivel criar o ambiente virtual.
    exit /b 1
)

"venv\Scripts\python.exe" -c "import struct,sys; raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')*8==64 else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERRO] O ambiente virtual criado nao usa um Python compativel de 64 bits.
    exit /b 1
)
exit /b 0


:PREPARAR_WEBVIEW2
echo Verificando Microsoft Edge WebView2 Runtime...
powershell -NoProfile -Command "$p=@('HKCU:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'); if($p.Where({Test-Path $_}).Count){exit 0}; exit 1" >nul 2>&1
if not errorlevel 1 (
    echo WebView2 Runtime encontrado.
    exit /b 0
)

where winget >nul 2>&1
if not errorlevel 1 (
    echo Instalando WebView2 Runtime para a interface nativa...
    winget install --id Microsoft.EdgeWebView2Runtime -e --silent --accept-package-agreements --accept-source-agreements
)

powershell -NoProfile -Command "$p=@('HKCU:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}','HKLM:\Software\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'); if($p.Where({Test-Path $_}).Count){exit 0}; exit 1" >nul 2>&1
if errorlevel 1 (
    echo [AVISO] WebView2 indisponivel; a interface usara o navegador padrao.
) else (
    echo WebView2 Runtime pronto.
)
exit /b 0


:PREPARAR_PYTORCH
echo Preparando PyTorch %TORCH_VERSION%...
if exist "%WINDIR%\System32\nvcuda.dll" (
    "%PY%" -c "import torch,torchaudio; clean=lambda v:v.split('+',1)[0]; ok=clean(torch.__version__)=='%TORCH_VERSION%' and clean(torchaudio.__version__)=='%TORCH_VERSION%' and torch.cuda.is_available(); raise SystemExit(0 if ok else 1)" >nul 2>&1
    if errorlevel 1 (
        echo NVIDIA detectada; instalando PyTorch CUDA 12.8...
        "%PY%" -m pip install --force-reinstall --retries 5 --timeout 90 "torch==%TORCH_VERSION%" "torchaudio==%TORCH_VERSION%" --index-url https://download.pytorch.org/whl/cu128
        if errorlevel 1 (
            echo [AVISO] Falha no pacote CUDA. Instalando fallback para CPU...
            "%PY%" -m pip install --force-reinstall --retries 5 --timeout 90 "torch==%TORCH_VERSION%" "torchaudio==%TORCH_VERSION%" --index-url https://download.pytorch.org/whl/cpu
        )
    ) else (
        echo PyTorch CUDA ja esta pronto.
    )
) else (
    "%PY%" -c "import torch,torchaudio; clean=lambda v:v.split('+',1)[0]; ok=clean(torch.__version__)=='%TORCH_VERSION%' and clean(torchaudio.__version__)=='%TORCH_VERSION%'; raise SystemExit(0 if ok else 1)" >nul 2>&1
    if errorlevel 1 (
        echo NVIDIA nao detectada; instalando PyTorch para CPU...
        "%PY%" -m pip install --force-reinstall --retries 5 --timeout 90 "torch==%TORCH_VERSION%" "torchaudio==%TORCH_VERSION%" --index-url https://download.pytorch.org/whl/cpu
    ) else (
        echo PyTorch ja esta pronto para CPU.
    )
)

if errorlevel 1 (
    echo [ERRO] Falha ao instalar PyTorch e torchaudio.
    exit /b 1
)

"%PY%" -c "import torch,torchaudio; print('PyTorch:',torch.__version__); print('CUDA disponivel:',torch.cuda.is_available())"
if errorlevel 1 (
    echo [ERRO] PyTorch ou torchaudio nao puderam ser importados.
    exit /b 1
)
exit /b 0


:INSTALAR_LLAMA
echo.
echo Baixando ou atualizando llama.cpp oficial [%LLAMA_CPP_BACKEND%]...
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\baixar_llama_cpp.ps1"
if errorlevel 1 (
    echo [ERRO] Falha ao baixar ou instalar llama.cpp.
    exit /b 1
)
if defined LLAMA_CPP_DIR (
    set "LLAMA_INSTALL_DIR=%LLAMA_CPP_DIR%"
) else (
    set "LLAMA_INSTALL_DIR=%CD%\llama.cpp"
)
if not exist "%LLAMA_INSTALL_DIR%\llama-server.exe" (
    echo [ERRO] llama-server.exe nao foi encontrado apos a instalacao.
    exit /b 1
)
"%LLAMA_INSTALL_DIR%\llama-server.exe" --version
if errorlevel 1 (
    echo [ERRO] llama-server.exe foi baixado, mas nao consegue iniciar.
    exit /b 1
)
exit /b 0


:LLAMA_CPP_ONLY
if not exist "scripts\baixar_llama_cpp.ps1" (
    echo [ERRO] scripts\baixar_llama_cpp.ps1 nao encontrado.
    goto FALHA
)
call :INSTALAR_LLAMA
if errorlevel 1 goto FALHA
goto SUCESSO


:CHECK_ONLY
if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual ausente. Execute instalar.bat primeiro.
    goto FALHA
)
"venv\Scripts\python.exe" "scripts\validar_instalacao.py" --strict
set "CHECK_RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %CHECK_RESULT%


:FALHA
echo.
echo ================================================
echo  Instalacao nao concluida
echo ================================================
echo Revise o erro acima e execute instalar.bat novamente.
pause
exit /b 1


:SUCESSO
echo.
pause
exit /b 0
