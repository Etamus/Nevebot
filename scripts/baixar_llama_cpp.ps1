param(
    [ValidateSet("auto", "cuda", "vulkan", "cpu")]
    [string] $Backend = $(if ($env:LLAMA_CPP_BACKEND) { $env:LLAMA_CPP_BACKEND } else { "auto" }),
    [string] $InstallDir = $(if ($env:LLAMA_CPP_DIR) { $env:LLAMA_CPP_DIR } else { "llama.cpp" }),
    [switch] $Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = [IO.Path]::GetFullPath((Join-Path $scriptDir ".."))

if ([IO.Path]::IsPathRooted($InstallDir)) {
    $installPath = [IO.Path]::GetFullPath($InstallDir)
} else {
    $installPath = [IO.Path]::GetFullPath((Join-Path $root $InstallDir))
}

$rootWithSlash = $root.TrimEnd("\") + "\"
$installWithSlash = $installPath.TrimEnd("\") + "\"
if (-not $installWithSlash.StartsWith($rootWithSlash, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Diretorio de instalacao invalido: $installPath"
}
if ($installPath.TrimEnd("\") -eq $root.TrimEnd("\")) {
    throw "Diretorio de instalacao nao pode ser a raiz do projeto."
}

$headers = @{ "User-Agent" = "Nevebot-installer" }
$releaseUrl = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"

Write-Host "Consultando ultima release em $releaseUrl ..."
$release = Invoke-RestMethod -Uri $releaseUrl -Headers $headers
$assets = @($release.assets | Where-Object { $_.name -like "*.zip" })
if (-not $assets) {
    throw "Nenhum asset .zip encontrado na release $($release.tag_name)."
}

function Get-CudaVersion {
    param([string] $Name)
    if ($Name -match "cuda-(?:cu)?([0-9]+(?:\.[0-9]+){0,2})") {
        return $Matches[1]
    }
    if ($Name -match "cuda(?:-)?cu([0-9]+(?:\.[0-9]+){0,2})") {
        return $Matches[1]
    }
    return ""
}

function Select-PreferredCudaAsset {
    param([array] $Candidates)
    $preferred = @("12.4", "12.5", "12.6", "12.8", "13.1", "13.0")
    foreach ($version in $preferred) {
        $found = @($Candidates | Where-Object { (Get-CudaVersion $_.name).StartsWith($version) }) | Select-Object -First 1
        if ($found) {
            return $found
        }
    }
    return @($Candidates | Sort-Object name -Descending) | Select-Object -First 1
}

function Test-NvidiaAvailable {
    $driverDll = Join-Path $env:WINDIR "System32\nvcuda.dll"
    return (Test-Path -LiteralPath $driverDll)
}

function Stop-LlamaProcessesInInstallDir {
    param([string] $Dir)

    $needle = [IO.Path]::GetFullPath($Dir).TrimEnd("\").ToLowerInvariant()
    $processes = @(
        Get-CimInstance Win32_Process -Filter "name = 'llama-server.exe' or name = 'llama-cli.exe' or name = 'llama.exe'" -ErrorAction SilentlyContinue |
        Where-Object {
            $exe = ([string]$_.ExecutablePath).ToLowerInvariant()
            $cmd = ([string]$_.CommandLine).ToLowerInvariant()
            $exe.StartsWith($needle) -or $cmd.Contains($needle)
        }
    )

    foreach ($proc in $processes) {
        Write-Warning "Encerrando processo llama.cpp em uso (PID $($proc.ProcessId)) para atualizar os binarios."
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }

    if ($processes.Count -gt 0) {
        Start-Sleep -Milliseconds 1200
    }
}

function Get-CudaBundle {
    $mainCandidates = @(
        $assets | Where-Object {
            $_.name -match "^llama-.*-bin-win-cuda.*-x64\.zip$" -and
            $_.name -notmatch "^cudart-"
        }
    )
    if (-not $mainCandidates) {
        throw "Asset CUDA do llama.cpp para Windows x64 nao encontrado."
    }

    $main = Select-PreferredCudaAsset $mainCandidates
    $version = Get-CudaVersion $main.name

    $runtimeCandidates = @(
        $assets | Where-Object {
            $_.name -match "^cudart-llama-bin-win-cuda.*-x64\.zip$"
        }
    )
    $runtime = @($runtimeCandidates | Where-Object { (Get-CudaVersion $_.name) -eq $version }) | Select-Object -First 1
    if (-not $runtime) {
        throw "Runtime CUDA correspondente ao asset $($main.name) nao encontrado."
    }

    return @{
        Backend = "cuda"
        Assets = @($main, $runtime)
    }
}

function Get-VulkanBundle {
    $main = @(
        $assets | Where-Object { $_.name -match "^llama-.*-bin-win-vulkan-x64\.zip$" }
    ) | Sort-Object name -Descending | Select-Object -First 1
    if (-not $main) {
        throw "Asset Vulkan do llama.cpp para Windows x64 nao encontrado."
    }
    return @{
        Backend = "vulkan"
        Assets = @($main)
    }
}

function Get-CpuBundle {
    $main = @(
        $assets | Where-Object { $_.name -match "^llama-.*-bin-win-(cpu|avx2|avx)-x64\.zip$" }
    ) | Sort-Object @{ Expression = { if ($_.name -match "cpu") { 0 } elseif ($_.name -match "avx2") { 1 } else { 2 } } }, name | Select-Object -First 1
    if (-not $main) {
        throw "Asset CPU do llama.cpp para Windows x64 nao encontrado."
    }
    return @{
        Backend = "cpu"
        Assets = @($main)
    }
}

$wanted = $Backend.ToLowerInvariant()
if ($wanted -eq "auto") {
    if (Test-NvidiaAvailable) {
        $wanted = "cuda"
    } else {
        $wanted = "cpu"
    }
}

try {
    if ($wanted -eq "cuda") {
        $bundle = Get-CudaBundle
    } elseif ($wanted -eq "vulkan") {
        $bundle = Get-VulkanBundle
    } else {
        $bundle = Get-CpuBundle
    }
} catch {
    if ($Backend.ToLowerInvariant() -eq "auto" -and $wanted -eq "cuda") {
        Write-Warning "$($_.Exception.Message) Usando asset CPU como fallback."
        $bundle = Get-CpuBundle
    } else {
        throw
    }
}

$serverPath = Join-Path $installPath "llama-server.exe"
$metadataPath = Join-Path $installPath "release.json"
$assetNames = @($bundle.Assets | ForEach-Object { $_.name })

if (-not $Force -and (Test-Path -LiteralPath $serverPath) -and (Test-Path -LiteralPath $metadataPath)) {
    try {
        $metadata = Get-Content -Raw -LiteralPath $metadataPath | ConvertFrom-Json
        $sameTag = ($metadata.tag -eq $release.tag_name)
        $sameBackend = ($metadata.backend -eq $bundle.Backend)
        $sameAssets = (@($metadata.assets) -join "|") -eq ($assetNames -join "|")
        if ($sameTag -and $sameBackend -and $sameAssets) {
            Write-Host "llama.cpp ja esta atualizado: $($release.tag_name) [$($bundle.Backend)]."
            exit 0
        }
    } catch {
        Write-Warning "Metadados antigos invalidos; reinstalando llama.cpp."
    }
}

$tempRoot = Join-Path $root "temp_llama"
$downloadDir = Join-Path $tempRoot "downloads"
$stageDir = Join-Path $tempRoot "stage"

New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
if (Test-Path -LiteralPath $stageDir) {
    Remove-Item -LiteralPath $stageDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null

foreach ($asset in $bundle.Assets) {
    $zipPath = Join-Path $downloadDir $asset.name
    Write-Host "Baixando $($asset.name) ..."
    Invoke-WebRequest -Uri $asset.browser_download_url -Headers $headers -OutFile $zipPath
    Write-Host "Extraindo $($asset.name) ..."
    Expand-Archive -LiteralPath $zipPath -DestinationPath $stageDir -Force
}

$serverInStage = Get-ChildItem -Path $stageDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $serverInStage) {
    throw "llama-server.exe nao foi encontrado nos assets baixados."
}

$payloadRoot = $stageDir
$stageChildren = @(Get-ChildItem -LiteralPath $stageDir)
if ($stageChildren.Count -eq 1 -and $stageChildren[0].PSIsContainer) {
    $payloadRoot = $stageChildren[0].FullName
}

if (Test-Path -LiteralPath $installPath) {
    Stop-LlamaProcessesInInstallDir $installPath
    try {
        Remove-Item -LiteralPath $installPath -Recurse -Force
    } catch {
        throw "Nao foi possivel substituir $installPath. Feche processos usando llama.cpp e tente novamente. Detalhe: $($_.Exception.Message)"
    }
}
New-Item -ItemType Directory -Force -Path $installPath | Out-Null
Copy-Item -Path (Join-Path $payloadRoot "*") -Destination $installPath -Recurse -Force

$finalServer = Join-Path $installPath "llama-server.exe"
if (-not (Test-Path -LiteralPath $finalServer)) {
    throw "Falha ao instalar llama-server.exe em $installPath."
}

$metadataOut = [ordered]@{
    source = "https://github.com/ggml-org/llama.cpp/releases/latest"
    tag = $release.tag_name
    backend = $bundle.Backend
    assets = $assetNames
    installed_at = (Get-Date).ToString("s")
}
$metadataOut | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $metadataPath -Encoding UTF8

Write-Host "llama.cpp instalado em $installPath"
Write-Host "Release: $($release.tag_name)"
Write-Host "Backend: $($bundle.Backend)"
