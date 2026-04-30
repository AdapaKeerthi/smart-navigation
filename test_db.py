"""
Run this in your project folder: python test_db.py
It will tell you exactly what's wrong with your database connection.
"""
import os
import sys

print("=" * 50)
print("DriveIQ Database Connection Test")
print("=" * 50)

# Step 1: Check dotenv
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ python-dotenv loaded")
except ImportError:
    print("❌ python-dotenv NOT installed")
    print("   Fix: pip install python-dotenv")
    sys.exit()

# Step 2: Check DATABASE_URL
url = os.environ.get("DATABASE_URL")
if not url:
    print("❌ DATABASE_URL not found in .env file")
    print("   Fix: Make sure your .env file exists and has DATABASE_URL=...")
    # Try to find .env file
    import pathlib
    env_files = list(pathlib.Path('.').glob('*.env')) + list(pathlib.Path('.').glob('.env'))
    if env_files:
        print(f"   Found these env files: {env_files}")
        for f in env_files:
            print(f"   Contents of {f}:")
            try:
                print("   " + open(f).read()[:200])
            except: pass
    else:
        print("   No .env files found in current folder!")
        print("   Current folder:", os.getcwd())
    sys.exit()

print(f"✅ DATABASE_URL found: {url[:40]}...")

# Step 3: Fix postgres:// → postgresql://
if url.startswith("postgres://"):
    url = url.replace("postgres://", "postgresql://", 1)
    print("✅ Fixed postgres:// → postgresql://")

# Step 4: Try connecting
try:
    import psycopg2
    print("✅ psycopg2 installed")
except ImportError:
    print("❌ psycopg2 NOT installed")
    print("   Fix: pip install psycopg2-binary")
    sys.exit()

print("\nConnecting to database...")
try:
    conn = psycopg2.connect(url, sslmode='require', connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    print(f"✅ Connected! PostgreSQL version: {version[:50]}")
    conn.close()
    print("\n✅ DATABASE IS WORKING! You can run python app.py now.")
except Exception as e:
    print(f"❌ Connection FAILED: {e}")
    print("\nCommon fixes:")
    print("  1. Check if your Render database is active (not suspended)")
    print("  2. Go to render.com → your DB → Copy the External Database URL again")
    print("  3. Make sure no quotes or spaces in the URL")

print("=" * 50)
