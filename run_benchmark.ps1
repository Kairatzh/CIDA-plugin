# PowerShell Script to run CIDA Latent Benchmark
Write-Host "=== Kairos-CIDA Benchmark Launcher ===" -ForegroundColor Cyan

# Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not in PATH. Please install Python 3.10+."
    exit 1
}

Write-Host "Installing/updating required packages (torch, transformers, datasets, huggingface_hub)..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install torch transformers datasets huggingface_hub torchdiffeq

# Check if CIDA plugin is installed, if not, install in editable mode
if (!(python -c "import cida_plugin" -ErrorAction SilentlyContinue)) {
    Write-Host "Installing CIDA-plugin in editable mode..." -ForegroundColor Yellow
    python -m pip install -e .
}

Write-Host "Running fast perplexity evaluation benchmark (fast_ppl_eval.py)..." -ForegroundColor Green
python fast_ppl_eval.py

Write-Host "=== Benchmark Finished ===" -ForegroundColor Cyan
