# PostgreSQL Setup on Windows (PowerShell Guide)

Complete step-by-step guide to install PostgreSQL, set up pgvector extension, and configure your database for the RAG pipeline.

---

## Step 1: Check if PostgreSQL is Already Installed

```powershell
# Check PostgreSQL version
psql --version

# If you see a version number, PostgreSQL is installed - skip to Step 3
# If you get "command not found", proceed to Step 2
```

---

## Step 2: Install PostgreSQL on Windows

### Option A: Download Installer (Recommended)

1. **Visit:** https://www.postgresql.org/download/windows/
2. **Click:** "Download the installer"
3. **Select:** PostgreSQL version 15 or 16 (latest stable)
4. **Download:** Windows x86-64 installer

**During Installation:**
- **Installation Directory:** Keep default (`C:\Program Files\PostgreSQL\16`)
- **Data Directory:** Keep default (`C:\Program Files\PostgreSQL\16\data`)
- **Password:** **SET A PASSWORD** - remember this! (e.g., `postgres123`)
- **Port:** Keep default `5432`
- **Advanced Options:** Keep defaults
- **Stack Builder:** Uncheck (not needed)

### Option B: Using Chocolatey (If you have it)

```powershell
# Install PostgreSQL
choco install postgresql --params '/Password:postgres123'

# PostgreSQL will auto-start as a service
```

### Verify Installation

```powershell
# Check if PostgreSQL service is running
Get-Service postgresql*

# Should show: Running
# If not running:
Start-Service postgresql*

# Check PostgreSQL version
psql --version
```

---

## Step 3: Get/Set Your PostgreSQL Password

### If You Just Installed PostgreSQL

Your password is the one you set during installation. **Remember it!**

### If PostgreSQL is Already Installed (Forgot Password)

**Option 1: Reset via Windows Authentication (Recommended)**

```powershell
# Open PowerShell as Administrator
# Right-click PowerShell → "Run as Administrator"

# Stop PostgreSQL service
Stop-Service postgresql*

# Edit pg_hba.conf file
notepad "C:\Program Files\PostgreSQL\16\data\pg_hba.conf"
```

**Find these lines and modify:**

```
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# IPv4 local connections:
host    all             all             127.0.0.1/32            md5
```

Change `md5` to `trust` temporarily:

```
host    all             all             127.0.0.1/32            trust
```

**Save and close.**

```powershell
# Start PostgreSQL
Start-Service postgresql*

# Connect without password
psql -U postgres

# Now reset your password
ALTER USER postgres WITH PASSWORD 'your_new_password';

# Exit
\q

# Edit pg_hba.conf again and change back to md5
notepad "C:\Program Files\PostgreSQL\16\data\pg_hba.conf"
# Change 'trust' back to 'md5'

# Restart PostgreSQL
Restart-Service postgresql*

# Test connection with new password
psql -U postgres
# Enter your new password when prompted
```

---

## Step 4: Test PostgreSQL Connection

```powershell
# Test connection
psql -U postgres -h localhost

# You'll be prompted for password - enter the one you set/remember

# If connected successfully, you'll see:
# postgres=#

# Type these commands:
\l                    # List all databases
\q                    # Quit
```

**If connection fails:**
1. Check if service is running: `Get-Service postgresql*`
2. Check firewall settings
3. Try `localhost` instead of IP

---

## Step 5: Install pgvector Extension

### Method 1: Download Pre-built Binaries (Easiest)

1. **Visit:** https://github.com/pgvector/pgvector/releases
2. **Download:** `windows-XX-x64-vector.zip` (match your PostgreSQL version)
   - For PostgreSQL 15: `windows-15-x64-vector.zip`
   - For PostgreSQL 16: `windows-16-x64-vector.zip`

3. **Extract the ZIP file**

4. **Copy files to PostgreSQL directories:**

```powershell
# Find your PostgreSQL installation path
$pgVersion = psql --version | Select-String -Pattern "(\d+)" | ForEach-Object { $_.Matches[0].Value }
$pgPath = "C:\Program Files\PostgreSQL\$pgVersion"

# Verify path exists
Test-Path $pgPath

# Copy vector.dll to lib folder
Copy-Item ".\vector.dll" -Destination "$pgPath\lib\"

# Copy vector.control to share\extension folder
Copy-Item ".\vector.control" -Destination "$pgPath\share\extension\"

# Copy *.sql files to share\extension folder
Copy-Item ".\vector--*.sql" -Destination "$pgPath\share\extension\"
```

### Method 2: Compile from Source

```powershell
# Install Visual Studio Build Tools first
# Download from: https://visualstudio.microsoft.com/downloads/
# Install "Desktop development with C++" workload

# Then:
# Install git if not already installed
winget install Git.Git

# Clone pgvector repository
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector

# Build (requires PostgreSQL development headers)
make

# Install
make install

# For Windows, you'll need to manually copy the compiled files
```

---

## Step 6: Enable pgvector Extension

```powershell
# Connect to PostgreSQL
psql -U postgres -h localhost

# Enter your password when prompted

# Enable the extension in the postgres database
CREATE EXTENSION vector;

# Verify it's installed
\dx

# You should see:
# Name    | Version |   Schema   |                   Description                    
#---------+---------+------------+-------------------------------------------------
# vector  | 0.7.0   | public     | vector data type and ivfflat access method

# Exit
\q
```

**If you get "extension vector does not exist":**
1. Check files were copied correctly
2. Restart PostgreSQL service: `Restart-Service postgresql*`
3. Check PostgreSQL logs: `C:\Program Files\PostgreSQL\16\data\log\`

---

## Step 7: Create RAG Database and Configure

```powershell
# Connect to PostgreSQL
psql -U postgres -h localhost

# Create database for RAG
CREATE DATABASE glaucoma_rag;

# Exit
\q

# Verify database was created
psql -U postgres -h localhost -c "\l" | Select-String "glaucoma_rag"
```

---

## Step 8: Configure Your Python Scripts

**Set your database password in environment variables:**

```powershell
# In PowerShell, set environment variable for current session
$env:DB_PASSWORD = "your_postgres_password"

# Verify it's set
echo $env:DB_PASSWORD

# To make it permanent (for all future sessions):
[Environment]::SetEnvironmentVariable("DB_PASSWORD", "your_postgres_password", "User")
```

**OR** edit the Python scripts directly:

Open `scripts/setup_postgres_vector_db.py` and `scripts/generate_and_store_embeddings.py`, find:

```python
DB_CONFIG = {
    "host": "localhost",
    "port": "5432",
    "user": "postgres",
    "password": "postgres",  # CHANGE THIS to your password
    "database": "postgres"
}
```

Change `"postgres"` to your actual password.

---

## Step 9: Test Everything Works

```powershell
# Test 1: PostgreSQL connection
psql -U postgres -h localhost -c "SELECT version();"

# Test 2: pgvector extension
psql -U postgres -h localhost -d postgres -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname = 'vector';"

# Test 3: Create test vector
psql -U postgres -h localhost -d postgres -c "SELECT '[1,2,3]'::vector AS test_vector;"

# Test 4: Run Python setup
python scripts/setup_postgres_vector_db.py

# Should output:
# ✓ Connected to PostgreSQL server
# ✓ Created database: glaucoma_rag
# ✓ Schema loaded successfully
# ✅ Database Setup Complete!
```

---

## Step 10: Troubleshooting Commands

```powershell
# Check PostgreSQL service status
Get-Service postgresql*

# Start PostgreSQL (if stopped)
Start-Service postgresql*

# Stop PostgreSQL
Stop-Service postgresql*

# Restart PostgreSQL
Restart-Service postgresql*

# Check PostgreSQL logs
Get-Content "C:\Program Files\PostgreSQL\16\data\log\*.log" -Tail 50

# Check if pgvector files exist
Test-Path "C:\Program Files\PostgreSQL\16\lib\vector.dll"
Test-Path "C:\Program Files\PostgreSQL\16\share\extension\vector.control"

# List all databases
psql -U postgres -h localhost -c "\l"

# List all extensions
psql -U postgres -h localhost -c "\dx"
```

---

## Common Issues and Solutions

### Issue: "psql: command not found"

**Solution:**
Add PostgreSQL to PATH:

```powershell
# Find PostgreSQL bin directory
$pgPath = "C:\Program Files\PostgreSQL\16\bin"

# Add to PATH for current session
$env:Path += ";$pgPath"

# Add permanently
[Environment]::SetEnvironmentVariable("Path", "$env:Path;$pgPath", "User")
```

### Issue: "password authentication failed"

**Solution:**
Reset password using Step 3 above.

### Issue: "connection refused"

**Solution:**
```powershell
# Check service is running
Get-Service postgresql*

# Start if stopped
Start-Service postgresql*

# Check port is not blocked
netstat -an | Select-String "5432"
```

### Issue: "extension vector does not exist"

**Solution:**
1. Verify files are copied correctly (Step 5)
2. Restart PostgreSQL: `Restart-Service postgresql*`
3. Try manually: `psql -U postgres -c "CREATE EXTENSION vector;"`

### Issue: "permission denied"

**Solution:**
Run PowerShell as Administrator:
```powershell
# Right-click PowerShell → "Run as Administrator"
```

---

## Quick Verification Checklist

```powershell
# Run these commands to verify everything is set up correctly:

# 1. PostgreSQL installed?
psql --version

# 2. Service running?
Get-Service postgresql*

# 3. Can connect?
psql -U postgres -c "SELECT version();"

# 4. pgvector installed?
psql -U postgres -c "\dx" | Select-String "vector"

# 5. Can create vectors?
psql -U postgres -c "SELECT '[1,2,3]'::vector;"

# 6. Environment variable set?
echo $env:DB_PASSWORD

# 7. Database created?
psql -U postgres -c "\l" | Select-String "glaucoma_rag"

# 8. Python can connect?
python -c "import psycopg2; conn = psycopg2.connect(host='localhost', user='postgres', password='$env:DB_PASSWORD', database='postgres'); print('✓ Python can connect!'); conn.close()"
```

---

## Ready to Run!

Once all checks pass, proceed with:

```powershell
# 1. Set up database
python scripts/setup_postgres_vector_db.py

# 2. Generate embeddings
python scripts/generate_and_store_embeddings.py

# 3. Test retrieval
python scripts/rag_retrieval.py
```

**All working? Proceed to Streamlit integration!** 🚀

