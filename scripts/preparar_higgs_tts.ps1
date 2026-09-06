param(
    [switch]$Check,
    [string]$Version = "v0.7.1"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$Root = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $Root "higgs.cpp"
$ModelDir = Join-Path $Root "models\higgs"
$DownloadDir = Join-Path $RuntimeDir ".downloads"
$ModelPath = Join-Path $ModelDir "higgs-audio-v3-tts-4b-q8_0.gguf"
$ModelSize = 5095354048L
$ModelSha256 = "79746822045b5bf8f9ab2bda87b16cd3f8ea3d9e319cbcf887a87aa1b537a74a"
$ReleaseBase = "https://github.com/0xShug0/audio.cpp/releases/download/$Version"
$ArchivePrefix = "audio-$Version"
$BinaryName = "$ArchivePrefix-bin-windows-x64-cuda13.3.zip"
$RuntimeName = "$ArchivePrefix-cudart-windows-x64-cuda13.3.zip"
$ModelUrl = "https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main/Higgs-Audio-v3-TTS-4B-GGUF/higgs-audio-v3-tts-4b-q8_0.gguf?download=true"

function Find-Server {
    if (-not (Test-Path -LiteralPath $RuntimeDir)) { return $null }
    return Get-ChildItem -LiteralPath $RuntimeDir -Filter "audiocpp_server.exe" -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
}

function Test-Model {
    if (-not (Test-Path -LiteralPath $ModelPath -PathType Leaf)) { return $false }
    return (Get-Item -LiteralPath $ModelPath).Length -eq $ModelSize
}

function Download-File([string]$Url, [string]$Destination, [long]$ExpectedSize = 0) {
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    if ((Test-Path -LiteralPath $Destination) -and (($ExpectedSize -le 0) -or ((Get-Item -LiteralPath $Destination).Length -eq $ExpectedSize))) {
        Write-Host "[OK] Download existente: $(Split-Path -Leaf $Destination)"
        return
    }

    Write-Host "Baixando $(Split-Path -Leaf $Destination)..."
    & curl.exe -L --fail --retry 8 --retry-all-errors --connect-timeout 30 -C - --output $Destination $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Falha no download de $Url (curl: $LASTEXITCODE)."
    }
    if ($ExpectedSize -gt 0 -and (Get-Item -LiteralPath $Destination).Length -ne $ExpectedSize) {
        throw "Download incompleto: $Destination"
    }
}

function Install-Archive([string]$Archive) {
    $staging = Join-Path $DownloadDir ([IO.Path]::GetFileNameWithoutExtension($Archive))
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $staging -Force
    Get-ChildItem -LiteralPath $staging -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $RuntimeDir -Recurse -Force
    }
    Remove-Item -LiteralPath $staging -Recurse -Force
}

function Assert-Ready([bool]$VerifyHash = $true) {
    $server = Find-Server
    if ($null -eq $server) { throw "audiocpp_server.exe nao encontrado em $RuntimeDir" }
    if (-not (Test-Model)) { throw "Modelo Higgs Q8_0 ausente ou incompleto em $ModelPath" }
    if ($VerifyHash) {
        $hash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $ModelSha256) { throw "Hash invalido para o modelo Higgs Q8_0." }
    }
    & $server.FullName --help *> $null
    if ($LASTEXITCODE -ne 0) { throw "audiocpp_server.exe nao consegue iniciar (codigo $LASTEXITCODE)." }
    Write-Host "[OK] Higgs TTS 3 Q8_0 pronto."
    Write-Host "     Runtime: $($server.FullName)"
    Write-Host "     Modelo:  $ModelPath"
}

if ($Check) {
    # A checagem de toda inicializacao usa tamanho exato; reler e calcular o
    # SHA256 de 5 GB atrasaria desnecessariamente a abertura da interface.
    Assert-Ready $false
    exit 0
}

New-Item -ItemType Directory -Force -Path $RuntimeDir, $ModelDir, $DownloadDir | Out-Null
$BinaryArchive = Join-Path $DownloadDir $BinaryName
$RuntimeArchive = Join-Path $DownloadDir $RuntimeName

Download-File "$ReleaseBase/$BinaryName" $BinaryArchive
Download-File "$ReleaseBase/$RuntimeName" $RuntimeArchive
Install-Archive $BinaryArchive
Install-Archive $RuntimeArchive
Download-File $ModelUrl $ModelPath $ModelSize

$hash = (Get-FileHash -LiteralPath $ModelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $ModelSha256) {
    throw "Hash invalido para o modelo Higgs Q8_0. Esperado $ModelSha256; recebido $hash."
}

Set-Content -LiteralPath (Join-Path $RuntimeDir ".version") -Value $Version -Encoding ASCII
Assert-Ready $true
