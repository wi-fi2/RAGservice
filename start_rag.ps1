# Start RAG Service - Ensures Ollama is running first

Write-Host "`n╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║          Starting RAG Service with Ollama LLM               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

# Step 1: Check if Ollama is installed
Write-Host "📦 Checking Ollama installation..." -ForegroundColor Yellow
try {
    $ollamaCheck = ollama list 2>&1
    Write-Host "✅ Ollama is installed`n" -ForegroundColor Green
} catch {
    Write-Host "❌ Ollama is not installed!" -ForegroundColor Red
    Write-Host "   Please install from: https://ollama.ai/download`n" -ForegroundColor Yellow
    pause
    exit
}

# Step 2: Start Ollama service
Write-Host "🚀 Starting Ollama service..." -ForegroundColor Yellow
try {
    # Check if Ollama is already running
    $ollamaTest = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -UseBasicParsing 2>$null
    Write-Host "✅ Ollama service already running`n" -ForegroundColor Green
} catch {
    Write-Host "   Starting Ollama serve..." -ForegroundColor Gray
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 3
    
    # Verify it started
    try {
        $ollamaTest = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 5 -UseBasicParsing
        Write-Host "✅ Ollama service started`n" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to start Ollama service!" -ForegroundColor Red
        Write-Host "   Try running: ollama serve`n" -ForegroundColor Yellow
        pause
        exit
    }
}

# Step 3: Check if model exists
Write-Host "🤖 Checking for Gemma 2 model..." -ForegroundColor Yellow
$models = ollama list | Out-String
if ($models -match "gemma2:2b") {
    Write-Host "✅ Gemma 2 2B model found`n" -ForegroundColor Green
} else {
    Write-Host "⚠️  Gemma 2 2B not found. Installing...`n" -ForegroundColor Yellow
    ollama pull gemma2:2b
}

# Step 4: Load environment variables
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]*)\s*=\s*(.*)$") {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Step 5: Activate virtual environment
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🐍 Activating Python virtual environment..." -ForegroundColor Yellow
    & "venv\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated`n" -ForegroundColor Green
}

# Step 6: Start the server
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "`n🚀 Starting RAG Server...`n" -ForegroundColor Green
Write-Host "📍 Server will be available at:" -ForegroundColor Cyan
Write-Host "   http://localhost:8000/static/index.html" -ForegroundColor White
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Run the server
python -m uvicorn app.server:app --host 0.0.0.0 --port 8000

