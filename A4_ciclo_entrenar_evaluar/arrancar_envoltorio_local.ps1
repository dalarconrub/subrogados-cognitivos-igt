<#
Anexo A, apartado A.4 · El ciclo completo: entrenar el adaptador y evaluar ensayo a ensayo.

Arranca el envoltorio local que reenvía las peticiones del cliente al servicio de inferencia y registra
su actividad.
#>

param(
    [int]$Port = 8080,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$RuntimeDir = Join-Path $RepoRoot "data_runtime\services"
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

$EnvFile = Join-Path $RepoRoot "infra\env\centaur.env.local"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) {
            return
        }
        $parts = $line.Split("=", 2)
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"').Trim("'")
        if ($key) {
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

$OutLog = Join-Path $LogDir "centaur_service.out.log"
$ErrLog = Join-Path $LogDir "centaur_service.err.log"

$Python = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$args = @(
    "-m", "uvicorn",
    "services.centaur_service:app",
    "--host", $HostAddress,
    "--port", "$Port",
    "--log-level", "info"
)

$process = Start-Process `
    -FilePath $Python `
    -ArgumentList $args `
    -WorkingDirectory $RepoRoot `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2

$PidFile = Join-Path $RuntimeDir "centaur_service.pid"
Set-Content -Path $PidFile -Value "$($process.Id)" -Encoding ASCII

Write-Output "centaur_service_pid=$($process.Id)"
Write-Output "pid_file=$PidFile"
Write-Output "health_url=http://${HostAddress}:${Port}/health"
Write-Output "generate_url=http://${HostAddress}:${Port}/generate"
Write-Output "stdout_log=$OutLog"
Write-Output "stderr_log=$ErrLog"
