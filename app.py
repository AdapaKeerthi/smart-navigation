from flask import Flask, render_template, request, jsonify, redirect, session
import os
import psycopg2
from model import model
from dotenv import load_dotenv
import bcrypt

load_dotenv()

app = Flask(__name__)
app.secret_key = "secret123"

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

    # admin user
    c.execute("""
    INSERT INTO users (id, username, password)
    VALUES (999,'admin','admin123')
    ON CONFLICT (id) DO NOTHING
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
        u = request.form['username']
        p = request.form['password']

        conn = get_db()
cur = conn.cursor()

cur.execute("SELECT id,password FROM users WHERE username=%s", (username,))
user = cur.fetchone()

if user and bcrypt.checkpw(password.encode(), user[1].encode()):
    session['user_id'] = user[0]
    return redirect('/map')

return "Invalid credentials"


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

            c.execute(
                "INSERT INTO users(username,password) VALUES(%s,%s)",
                hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt())
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


# ================= SAVE BEHAVIOR (FIXED) =================
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

        # ✅ SAFE MODEL CALL
        try:
            prediction = model.predict([[d, s, c]])
            driver_type = prediction[0]
        except Exception as e:
            print("MODEL ERROR:", e)
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


# ================= LIVE TRACKING (FIXED) =================
@app.route('/live_data')
def live_data():

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT users.username,
           driving_data.score,
           driving_data.latitude,
           driving_data.longitude
    FROM driving_data
    JOIN users ON users.id = driving_data.user_id
    WHERE driving_data.timestamp > NOW() - INTERVAL '2 minutes'
    ORDER BY driving_data.timestamp DESC
    """)

    rows = cur.fetchall()
    conn.close()

    data = []

    for r in rows:
        data.append({
            "username": r[0],
            "score": r[1],
            "lat": r[2],
            "lon": r[3]
        })

    return jsonify(data)


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


# ================= ADMIN =================
@app.route('/admin')
def admin():
    if 'user_id' not in session or session['user_id'] != 999:
        return "Access Denied"

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id, username FROM users")
    users = cur.fetchall()

    conn.close()

    return render_template('admin.html', users=users)


# ================= USER DETAIL =================
@app.route('/user/<username>')
def user_detail(username):

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username=%s",(username,))
    user = cur.fetchone()

    if not user:
        return "User not found"

    cur.execute("""
    SELECT deviations, stops, confusion, score, driver_type, timestamp
    FROM driving_data
    WHERE user_id=%s
    ORDER BY timestamp DESC
    """,(user[0],))

    data = cur.fetchall()
    conn.close()

    return render_template("user_detail.html", username=username, data=data)


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)