<#
.SYNOPSIS
    Set up and drive the DARWIN budget-free full-run test (TEST_RUN_PLAN.md).

.DESCRIPTION
    Builds the one slim darwin-agent image (serves all three container roles in the test profile),
    ensures the darwin-egress network exists, optionally resets prior run state, then runs the
    generational loop against run.test.yaml and prints the dashboard.

    Default is the $0 mock run (no API key needed). Pass -RealClaude to opt into the real,
    wall-clock-capped Claude mutator + global pass (needs $env:ANTHROPIC_API_KEY).

.EXAMPLE
    ./scripts/test_run.ps1                       # default mock run, 3 generations
    ./scripts/test_run.ps1 -Generations 2 -Fresh # wipe prior state, run 2 generations
    ./scripts/test_run.ps1 -RealClaude -Generations 1
    ./scripts/test_run.ps1 -DashboardOnly        # just print the dashboard for the current runs/
#>
[CmdletBinding()]
param(
    [int]$Generations = 3,
    [string]$Config = "run.test.yaml",
    [switch]$Fresh,          # delete models/ runs/ memory(model dirs) before running (clean slate)
    [switch]$SkipBuild,      # don't rebuild the image (reuse an existing darwin-agent)
    [switch]$RealClaude,     # set mutation.backend via env + real_global_memory (needs API key)
    [switch]$DashboardOnly   # skip setup + run; just render the dashboard
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

function Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# A stale PYTHONHOME/PYTHONPATH (left by an old Python install or a prior `activate`) poisons EVERY
# interpreter launch — the venv python then resolves the wrong home and dies with
# "did not find executable at ...python.exe". Clear them for this process so the run is hermetic.
foreach ($v in @("PYTHONHOME", "PYTHONPATH")) {
    if (Test-Path "Env:$v") { Write-Host "clearing stale $v=$((Get-Item Env:$v).Value)" -ForegroundColor DarkYellow; Remove-Item "Env:$v" }
}

# Resolve the project venv's Python and call it DIRECTLY (not via `uv run`). `uv run` re-resolves
# its managed interpreter on every invocation, which can transiently fail with
# "did not find executable at ...\uv\python\..."; calling the venv python directly is stabler.
# We also VERIFY the interpreter actually launches (the venv python is a trampoline to a uv-managed
# base interpreter; if that base was removed/quarantined, the trampoline exists but won't run), and
# repair via `uv sync` / `uv python install --reinstall` when it doesn't.
# Launch the interpreter without letting a native failure terminate the script (so the caller can
# branch on it). Returns $true iff `python -c pass` exits 0.
function Test-PyLaunch($py) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $py -c "pass" 2>$null | Out-Null; return ($LASTEXITCODE -eq 0) }
    catch { return $false }
    finally { $ErrorActionPreference = $prev }
}

function Get-VenvPython {
    $py = Join-Path $repo ".venv\Scripts\python.exe"
    if ((-not (Test-Path $py)) -or (-not (Test-PyLaunch $py))) {
        Step "Repairing Python interpreter (reinstall managed base + fresh venv)"
        $cfg = Join-Path $repo ".venv\pyvenv.cfg"
        $ver = ""
        if (Test-Path $cfg) {
            $m = Select-String -Path $cfg -Pattern '^version_info = (.+)$'
            if ($m) { $ver = $m.Matches.Groups[1].Value }
        }
        if ($ver) { uv python install --reinstall $ver }
        if (Test-Path (Join-Path $repo ".venv")) { Remove-Item -Recurse -Force (Join-Path $repo ".venv") }
        uv sync; if ($LASTEXITCODE -ne 0) { throw "uv sync failed." }
        if (-not (Test-PyLaunch $py)) {
            throw @"
The project's Python at $py still won't launch ('cannot find the path specified').
This is an environment problem, not a DARWIN bug. Most likely one of:
  - %APPDATA% / profile-folder redirection (corporate or OneDrive Known-Folder-Move) so
    C:\Users\<you>\AppData\Roaming\uv\python\... isn't resolvable in your shell;
  - antivirus quarantine of the uv-managed interpreter;
  - a stale PYTHONHOME/PYTHONPATH.
Diagnose by running the base interpreter directly:
  & 'C:\Users\$env:USERNAME\AppData\Roaming\uv\python\cpython-$ver-windows-x86_64-none\python.exe' --version
If that fails too, point uv's Python dir at a local, non-redirected path:
  setx UV_PYTHON_INSTALL_DIR C:\uv-python ; (open a NEW shell) ; uv sync
"@
        }
    }
    return $py
}

$Py = Get-VenvPython

if ($DashboardOnly) {
    & $Py -m darwin.observability --runs runs --cost runs/cost.jsonl
    return
}

# 0. preconditions ---------------------------------------------------------------
Step "Checking Docker"
docker run --rm hello-world | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker is not working (need Docker Desktop, Linux-container mode)." }

if ($RealClaude -and -not $env:ANTHROPIC_API_KEY) {
    throw "-RealClaude requires `$env:ANTHROPIC_API_KEY to be set."
}

# 1. build the slim image (serves agent + mock finetune + mock eval) -------------
if (-not $SkipBuild) {
    Step "Building darwin-agent image"
    docker build -f containers/darwin-agent.Dockerfile --build-arg HARNESS=agent -t darwin-agent .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed." }
} else {
    Step "Skipping image build (-SkipBuild)"
}

# 2. ensure the egress network exists --------------------------------------------
Step "Ensuring darwin-egress network"
$net = docker network ls --filter "name=^darwin-egress$" --format "{{.Name}}"
if ($net -ne "darwin-egress") {
    docker network create darwin-egress | Out-Null
    Write-Host "created darwin-egress"
} else {
    Write-Host "darwin-egress already exists"
}

# 3. optional clean slate --------------------------------------------------------
if ($Fresh) {
    Step "Resetting prior run state (-Fresh)"
    # Clear generated contents but preserve the tracked .gitkeep placeholders (so the dirs stay).
    foreach ($d in @("models", "runs", "eval_slices")) {
        if (Test-Path $d) {
            Get-ChildItem -Force $d | Where-Object { $_.Name -ne ".gitkeep" } |
                Remove-Item -Recurse -Force
        }
    }
    if (Test-Path "memory/models") { Remove-Item -Recurse -Force "memory/models" }
    # Note: memory/global/*.md (the seeded global store) is rewritten in place by each run; if you
    # want to reset it to the committed seeds, run `git checkout -- memory/global`.
}

# 4. run the loop ----------------------------------------------------------------
Step "Running $Generations generation(s) from $Config"
if ($RealClaude) {
    Write-Host "Real-Claude mode: set mutation.backend=claude (or claude_sample) + agent_network=open + real_global_memory=true in $Config." -ForegroundColor Yellow
    $env:DARWIN_REAL_CLAUDE = "1"
}
# Call the venv python directly (see Get-VenvPython): avoids both the stale `darwin` console-script
# shim and `uv run`'s transient managed-interpreter resolution error.
& $Py -m darwin --config $Config --generations $Generations
if ($LASTEXITCODE -ne 0) { throw "darwin run exited $LASTEXITCODE." }

# 5. dashboard -------------------------------------------------------------------
Step "Dashboard"
& $Py -m darwin.observability --runs runs --cost runs/cost.jsonl

Step "Containers that ran (ephemeral --rm; look for darwin-agent-* / darwin-finetune-* / darwin-eval-*)"
docker ps -a --filter "name=darwin-" --format "table {{.Names}}\t{{.Status}}" | Select-Object -First 20
