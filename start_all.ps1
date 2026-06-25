Write-Host "Starting AI Perception Project..." -ForegroundColor Green

Write-Host "1. Starting Backend, Frontend, and MLflow (Docker)..." -ForegroundColor Cyan
docker-compose up -d

Write-Host "`nAll core modules (Frontend, Backend Pipeline, MLflow) have been initiated!" -ForegroundColor Green
Write-Host "----------------------------------------------------"
Write-Host "Frontend Dashboard: http://localhost:5173"
Write-Host "Backend API:        http://localhost:8000"
Write-Host "MLflow UI:          http://localhost:5000"
Write-Host "----------------------------------------------------"

Read-Host "Press Enter to stop Docker containers and exit"
Write-Host "Stopping Docker containers..."
docker-compose down
Write-Host "Done." -ForegroundColor Green
