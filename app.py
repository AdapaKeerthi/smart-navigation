from flask import Flask, render_template, request, jsonify, redirect, session
import os
import psycopg2
from model import model
from dotenv import load_dotenv
import bcrypt

load_dotenv()

app = Flask(__name__)

# 🔥 SESSION FIX (IMPORTANT FOR RENDER)
app.secret_key = "super_secret_key_123"
app.config.update(
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None"
)

# ================= DB =================
def get_db():
    DATABASE_URL = os.environ.get("DATABASE_URL")

    if not DATABASE_URL:
        print("❌ DATABASE_URL missing")
        return None

    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

    try:
        return psycopg2.connect(DATABASE_URL, sslmode='require')
    except Exception as e:
        print("❌ DB CONNECTION ERROR:", e)
        return None


def init_db():
    conn = get_db()
    if not conn:
        return

    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS driving_data(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        deviations INTEGER,
        stops INTEGER,
        confusion INTEGER,
        score INTEGER,
        driver_type TEXT,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ================= ROUTES =================
@app.route('/')
def home():
    return redirect('/login')


@app.route('/map')
def map_page():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('index.html')


# ================= LOGIN =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()
        if not conn:
            return "DB error"

        cur = conn.cursor()

        cur.execute("SELECT id, password FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        conn.close()

        if user:
            stored_password = user[1]

            # 🔥 supports both plain + hashed
            if stored_password == password or bcrypt.checkpw(password.encode(), stored_password.encode()):
                session['user_id'] = user[0]
                session.permanent = True
                return redirect('/map')

        return "Invalid credentials"

    return render_template('login.html')


# ================= REGISTER =================
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']

        conn = get_db()
        if not conn:
            return "DB error"

        try:
            c = conn.cursor()

            c.execute("SELECT id FROM users WHERE username=%s", (u,))
            if c.fetchone():
                return "Username already exists"

            # 🔥 FIXED HASH
            hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()

            c.execute(
                "INSERT INTO users(username,password) VALUES(%s,%s)",
                (u, hashed)
            )

            conn.commit()
            conn.close()

            return redirect('/login')

        except Exception as e:
            print("REGISTER ERROR:", e)
            return "Registration failed"

    return render_template('register.html')


# ============= Logout ====================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ================= SAVE BEHAVIOR =================
@app.route('/save_behavior', methods=['POST'])
def save_behavior():
    if 'user_id' not in session:
        return jsonify({"error": "login required"})

    try:
        data = request.json

        d = data.get('deviations', 0)
        s = data.get('stops', 0)
        c = data.get('confusion', 0)
        lat = data.get('lat', 0)
        lon = data.get('lon', 0)

        score = max(0, 100 - (d*5 + s*3 + c*4))

        try:
            prediction = model.predict([[d, s, c]])
            driver_type = prediction[0]
        except:
            driver_type = "unknown"

        conn = get_db()
        if not conn:
            return jsonify({"error": "DB error"})

        cur = conn.cursor()

        cur.execute("""
        INSERT INTO driving_data(
            user_id,deviations,stops,confusion,
            score,driver_type,latitude,longitude
        )
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
        """, (session['user_id'], d, s, c, score, driver_type, lat, lon))

        conn.commit()
        conn.close()

        return jsonify({"score": score, "driver_type": driver_type})

    except Exception as e:
        print("SAVE ERROR:", e)
        return jsonify({"error": "save failed"})


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT deviations, stops, confusion, score, driver_type, timestamp
    FROM driving_data
    WHERE user_id=%s
    ORDER BY timestamp DESC
    """,(session['user_id'],))

    data = cur.fetchall()
    conn.close()

    return render_template('dashboard.html', data=data)


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)