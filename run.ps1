# Rail Saarthi - Start server (run after: pip install -r requirements.txt, then cd backend; python train.py)
Set-Location $PSScriptRoot
if (-not (Test-Path "backend\models\category_classifier.joblib")) {
    Write-Host "Models not found. Training first..." -ForegroundColor Yellow
    Set-Location backend
    python train.py
    Set-Location ..
}
Write-Host "Starting Rail Saarthi at http://localhost:8000" -ForegroundColor Green
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
