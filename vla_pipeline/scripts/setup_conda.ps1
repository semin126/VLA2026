param([string]$CondaExe = '')
$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
if (-not $CondaExe) {
    $found = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($found) { $CondaExe = $found.Source }
    foreach ($distribution in @('miniforge3', 'miniconda3', 'anaconda3')) {
        $candidate = Join-Path $env:USERPROFILE "$distribution\Scripts\conda.exe"
        if (-not $CondaExe -and (Test-Path -LiteralPath $candidate)) { $CondaExe = $candidate }
    }
}
if (-not $CondaExe) { throw 'Install Miniforge/conda or provide -CondaExe with the path to conda.exe.' }
$gitCommand = Get-Command git.exe -ErrorAction SilentlyContinue
if (-not $gitCommand) {
    $gitDirectory = Join-Path $env:ProgramFiles 'Git\cmd'
    if (Test-Path -LiteralPath (Join-Path $gitDirectory 'git.exe')) { $env:Path = "$gitDirectory;$env:Path" }
}
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first.' }
$environments = (& $CondaExe env list --json | ConvertFrom-Json).envs
$target = $environments | Where-Object { (Split-Path -Leaf $_) -eq 'vla2026' } | Select-Object -First 1
if (-not $target) {
    & $CondaExe create -n vla2026 --override-channels -c conda-forge python=3.12.16 pip --yes
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
    $target = ((& $CondaExe env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path -Leaf $_) -eq 'vla2026' } | Select-Object -First 1)
}
$python = Join-Path $target 'python.exe'
& $python -m pip install uv==0.13.7
if ($LASTEXITCODE -ne 0) { throw 'uv installation failed.' }
& $python -m uv pip install --python $python -r (Join-Path $project 'requirements-windows-ml-lock.txt') --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) { throw 'ML package installation failed.' }
& $python -m ipykernel install --user --name vla2026 --display-name 'VLA2026 (conda, Python 3.12, CUDA)'
if ($LASTEXITCODE -ne 0) { throw 'Jupyter kernel registration failed.' }
$env:PYTHONIOENCODING = 'utf-8'
& $python (Join-Path $project 'scripts/check_env.py')
if ($LASTEXITCODE -ne 0) { throw 'Environment check failed.' }
Write-Output "VS Code notebook kernel: VLA2026 (conda, Python 3.12, CUDA)"
Write-Output "Python: $python"
