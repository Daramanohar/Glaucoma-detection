# Quick Commands to Enable pgvector

Copy and paste these commands in your PowerShell terminal:

---

## If You're Inside psql (see "postgres-#" prompt):

Just type:
```
\q
```
Press Enter. This exits psql and returns you to PowerShell.

---

## Then Run These Commands in PowerShell:

### 1. Connect to PostgreSQL and Enable pgvector:

```powershell
# Connect and enable extension (will prompt for password)
psql -U postgres -c "CREATE EXTENSION vector;"
```

### 2. Verify pgvector is Enabled:

```powershell
# Check extensions
psql -U postgres -c "\dx"
```

You should see:
```
Name    | Version |   Schema   |                   Description                    
---------+---------+------------+-------------------------------------------------
vector  | 0.7.0   | public     | vector data type and ivfflat access method
```

### 3. Test Vector Creation:

```powershell
# Test creating a vector
psql -U postgres -c "SELECT '[1,2,3]'::vector AS test_vector;"
```

Should output:
```
 test_vector 
-------------
 [1,2,3]
(1 row)
```

---

## If You Get "extension vector does not exist":

This means pgvector files weren't copied correctly. Follow these steps:

### Step 1: Find PostgreSQL Version

```powershell
psql --version
# Look for the version number (e.g., "PostgreSQL 16.1")
```

### Step 2: Download pgvector for Your Version

1. Visit: https://github.com/pgvector/pgvector/releases
2. Find: `windows-XV-x64-vector.zip` where XV = your version (15 or 16)
3. Download and extract

### Step 3: Copy Files to PostgreSQL

```powershell
# Replace "16" with your version number
$pgPath = "C:\Program Files\PostgreSQL\16"

# Copy files (adjust paths to where you extracted)
Copy-Item ".\vector.dll" -Destination "$pgPath\lib\" -Force
Copy-Item ".\vector.control" -Destination "$pgPath\share\extension\" -Force
Copy-Item ".\vector--*.sql" -Destination "$pgPath\share\extension\" -Force
```

### Step 4: Restart PostgreSQL

```powershell
Restart-Service postgresql*
```

### Step 5: Enable Extension Again

```powershell
psql -U postgres -c "CREATE EXTENSION vector;"
```

---

## Complete Setup Workflow:

```powershell
# 1. Exit psql if you're inside it
# Type: \q   (if you see postgres-#)

# 2. Set your password (optional, if you want to avoid password prompts)
$env:PGPASSWORD = "your_password"

# 3. Enable pgvector
psql -U postgres -c "CREATE EXTENSION vector;"

# 4. Verify
psql -U postgres -c "\dx" | Select-String "vector"

# 5. Test
psql -U postgres -c "SELECT '[1,2,3]'::vector;"

# 6. Create RAG database
psql -U postgres -c "CREATE DATABASE glaucoma_rag;"

# 7. Enable pgvector in RAG database
psql -U postgres -d glaucoma_rag -c "CREATE EXTENSION vector;"

# 8. Verify in RAG database
psql -U postgres -d glaucoma_rag -c "\dx"
```

---

## Quick Test Everything Works:

```powershell
# Run the verification script
.\scripts\verify_postgres_setup.ps1

# Should show: ✓ pgvector extension installed
```

---

## What to Do Right Now:

**In your current psql session:**

1. Type: `\q` (quit psql)
2. In PowerShell, run:
   ```powershell
   psql -U postgres -c "CREATE EXTENSION vector;"
   ```

That's it! ✅

