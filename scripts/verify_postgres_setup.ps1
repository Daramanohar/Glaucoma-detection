# PowerShell Script to Verify PostgreSQL + pgvector Setup
# Run: .\scripts\verify_postgres_setup.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PostgreSQL + pgvector Setup Verification" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$allChecksPassed = $true

# Check 1: PostgreSQL installed
Write-Host "`n[1] Checking PostgreSQL installation..." -ForegroundColor Yellow
try {
    $pgVersion = psql --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ PostgreSQL installed: $pgVersion" -ForegroundColor Green
    } else {
        Write-Host "  ✗ PostgreSQL not found in PATH" -ForegroundColor Red
        Write-Host "    Install from: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        $allChecksPassed = $false
    }
} catch {
    Write-Host "  ✗ PostgreSQL not installed" -ForegroundColor Red
    $allChecksPassed = $false
}

# Check 2: PostgreSQL service running
Write-Host "`n[2] Checking PostgreSQL service..." -ForegroundColor Yellow
try {
    $service = Get-Service postgresql* -ErrorAction SilentlyContinue
    if ($service -and $service.Status -eq "Running") {
        Write-Host "  ✓ PostgreSQL service is running" -ForegroundColor Green
    } else {
        Write-Host "  ✗ PostgreSQL service not running" -ForegroundColor Red
        Write-Host "    Start with: Start-Service postgresql*" -ForegroundColor Yellow
        $allChecksPassed = $false
    }
} catch {
    Write-Host "  ✗ Could not check PostgreSQL service" -ForegroundColor Red
    $allChecksPassed = $false
}

# Check 3: Test database connection
Write-Host "`n[3] Testing database connection..." -ForegroundColor Yellow
try {
    $dbPassword = $env:DB_PASSWORD
    if (-not $dbPassword) {
        Write-Host "  ⚠ DB_PASSWORD not set in environment" -ForegroundColor Yellow
        Write-Host "    Set with: `$env:DB_PASSWORD = 'your_password'" -ForegroundColor Yellow
    }
    
    # Try to connect (will prompt for password if DB_PASSWORD not set)
    $result = psql -U postgres -h localhost -c "SELECT 1;" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Can connect to PostgreSQL" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Cannot connect to PostgreSQL" -ForegroundColor Red
        Write-Host "    Check: password, service status, firewall" -ForegroundColor Yellow
        $allChecksPassed = $false
    }
} catch {
    Write-Host "  ✗ Connection test failed" -ForegroundColor Red
    $allChecksPassed = $false
}

# Check 4: pgvector extension
Write-Host "`n[4] Checking pgvector extension..." -ForegroundColor Yellow
try {
    $result = psql -U postgres -h localhost -c "SELECT extversion FROM pg_extension WHERE extname = 'vector';" 2>&1
    if ($LASTEXITCODE -eq 0 -and $result -match "0\.\d+\.\d+") {
        Write-Host "  ✓ pgvector extension installed: $($result[0])" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ pgvector extension not enabled" -ForegroundColor Yellow
        Write-Host "    Enable with: psql -U postgres -c 'CREATE EXTENSION vector;'" -ForegroundColor Yellow
        $allChecksPassed = $false
    }
} catch {
    Write-Host "  ✗ pgvector check failed" -ForegroundColor Red
    $allChecksPassed = $false
}

# Check 5: pgvector files exist
Write-Host "`n[5] Checking pgvector files..." -ForegroundColor Yellow
$pgVersions = @("16", "15", "14", "13", "12")
$found = $false

foreach ($version in $pgVersions) {
    $vectorDll = "C:\Program Files\PostgreSQL\$version\lib\vector.dll"
    $vectorControl = "C:\Program Files\PostgreSQL\$version\share\extension\vector.control"
    
    if ((Test-Path $vectorDll) -and (Test-Path $vectorControl)) {
        Write-Host "  ✓ pgvector files found for PostgreSQL $version" -ForegroundColor Green
        $found = $true
        break
    }
}

if (-not $found) {
    Write-Host "  ✗ pgvector files not found" -ForegroundColor Red
    Write-Host "    Download from: https://github.com/pgvector/pgvector/releases" -ForegroundColor Yellow
    $allChecksPassed = $false
}

# Check 6: Database exists
Write-Host "`n[6] Checking glaucoma_rag database..." -ForegroundColor Yellow
try {
    $result = psql -U postgres -h localhost -lqt | Select-String "glaucoma_rag"
    if ($result) {
        Write-Host "  ✓ glaucoma_rag database exists" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ glaucoma_rag database not found" -ForegroundColor Yellow
        Write-Host "    Will be created by: python scripts/setup_postgres_vector_db.py" -ForegroundColor Yellow
    }
} catch {
    Write-Host "  ✗ Could not check databases" -ForegroundColor Red
}

# Check 7: Python dependencies
Write-Host "`n[7] Checking Python dependencies..." -ForegroundColor Yellow
$pythonPackages = @("psycopg2", "sentence_transformers", "tiktoken")

foreach ($package in $pythonPackages) {
    try {
        python -c "import $package" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  ✓ $package installed" -ForegroundColor Green
        } else {
            Write-Host "  ✗ $package not installed" -ForegroundColor Red
            Write-Host "    Install with: pip install $package" -ForegroundColor Yellow
            $allChecksPassed = $false
        }
    } catch {
        Write-Host "  ✗ $package check failed" -ForegroundColor Red
        $allChecksPassed = $false
    }
}

# Summary
Write-Host "`n============================================================" -ForegroundColor Cyan
if ($allChecksPassed) {
    Write-Host "✅ All checks passed! Ready to proceed." -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. python scripts/setup_postgres_vector_db.py" -ForegroundColor White
    Write-Host "  2. python scripts/generate_and_store_embeddings.py" -ForegroundColor White
    Write-Host "  3. python scripts/rag_retrieval.py" -ForegroundColor White
} else {
    Write-Host "⚠ Some checks failed. Please fix the issues above." -ForegroundColor Red
    Write-Host "`nSee WINDOWS_POSTGRES_SETUP.md for detailed instructions." -ForegroundColor Yellow
}
Write-Host "============================================================" -ForegroundColor Cyan

