# PowerShell Script to Reset RAG Database
# Use this if you need to start fresh
# Run: .\scripts\reset_rag_database.ps1

Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "RAG Database Reset Script" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow

Write-Host "`n⚠ WARNING: This will delete all data from glaucoma_rag database!" -ForegroundColor Red
$confirmation = Read-Host "Are you sure? Type 'yes' to continue"

if ($confirmation -ne "yes") {
    Write-Host "`nReset cancelled." -ForegroundColor Yellow
    exit
}

# Get database password
$dbPassword = $env:DB_PASSWORD
if (-not $dbPassword) {
    $dbPassword = Read-Host "Enter PostgreSQL password"
}

Write-Host "`nResetting database..." -ForegroundColor Yellow

# Connect and drop database
try {
    $env:PGPASSWORD = $dbPassword
    
    # Drop and recreate database
    Write-Host "[1] Dropping existing database..." -ForegroundColor Cyan
    psql -U postgres -h localhost -c "DROP DATABASE IF EXISTS glaucoma_rag;" 2>&1 | Out-Null
    
    Write-Host "[2] Creating fresh database..." -ForegroundColor Cyan
    psql -U postgres -h localhost -c "CREATE DATABASE glaucoma_rag;" 2>&1 | Out-Null
    
    Write-Host "[3] Enabling pgvector extension..." -ForegroundColor Cyan
    psql -U postgres -h localhost -d glaucoma_rag -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1 | Out-Null
    
    Write-Host "`n✅ Database reset complete!" -ForegroundColor Green
    
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. python scripts/setup_postgres_vector_db.py" -ForegroundColor White
    Write-Host "  2. python scripts/generate_and_store_embeddings.py" -ForegroundColor White
    
} catch {
    Write-Host "`n❌ Error resetting database: $_" -ForegroundColor Red
    Write-Host "Please check PostgreSQL is running and credentials are correct." -ForegroundColor Yellow
} finally {
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
}

Write-Host "============================================================" -ForegroundColor Yellow

