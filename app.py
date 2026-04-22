from flask import Flask, render_template, request, jsonify, redirect, session
import os
import psycopg2
from model import model
from dotenv import load_dotenv

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
        print("❌ Skipping DB init")
        return

    try:
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

        c.execute("""
        INSERT INTO users (id, username, password)
        VALUES (999,'admin','admin123')
        ON CONFLICT (id) DO NOTHING
        """)

        conn.commit()
        conn.close()

        print("✅ DB initialized")

    except Exception as e:
        print("❌ INIT DB ERROR:", e)


# ⚠️ IMPORTANT: DO NOT crash app if DB fails
try:
    init_db()
except:
    print("⚠️ DB skipped during startup")


# ================= ROUTES =================
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db()   # ✅ FIXED
        if not conn:
            return "DB error"

        cur = conn.cursor()

        cur.execute(
            "SELECT id FROM users WHERE username=%s AND password=%s",  # ✅ FIXED
            (username, password)
        )

        user = cur.fetchone()
        conn.close()

        if user:
            session['user_id'] = user[0]   # ✅ FIXED (session set)
            return redirect('/')
        else:
            return "Invalid credentials"

    return render_template('login.html')


@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']

        try:
            conn = get_db()
            if not conn:
                return "DB connection failed"

            c = conn.cursor()

            c.execute(
                "INSERT INTO users(username,password) VALUES(%s,%s)",
                (u,p)
            )

            conn.commit()
            conn.close()

            return redirect('/login')

        except Exception as e:
            print("REGISTER ERROR:", e)
            return "Registration failed"

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ================= SAVE =================
@app.route('/save_behavior', methods=['POST'])
def save_behavior():
    if 'user_id' not in session:
        return jsonify({"error":"login required"})

    data = request.json

    d = data.get('deviations', 0)
    s = data.get('stops', 0)
    c = data.get('confusion', 0)
    lat = data.get('lat', 0)
    lon = data.get('lon', 0)

    score = max(0, 100 - (d*5 + s*3 + c*4))

    prediction = model.predict([[d,s,c]])
    driver_type = prediction[0]

    conn = get_db()
    if not conn:
        return jsonify({"error":"DB error"})   # ✅ FIXED

    cur = conn.cursor()

    cur.execute("""
    INSERT INTO driving_data(user_id,deviations,stops,confusion,score,driver_type,latitude,longitude)
    VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
    """,(session['user_id'],d,s,c,score,driver_type,lat,lon))

    conn.commit()
    conn.close()

    return jsonify({"score":score,"driver_type":driver_type})


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


# ================= LIVE DATA =================
@app.route('/live_data')
def live_data():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT DISTINCT ON (users.username)
        users.username,
        driving_data.score,
        driving_data.latitude,
        driving_data.longitude
    FROM driving_data
    JOIN users ON users.id = driving_data.user_id
    ORDER BY users.username, driving_data.timestamp DESC
    """)

    rows = cur.fetchall()
    conn.close()

    data = []

    for r in rows:
        data.append({
            "username": r[0],
            "score": r[1],
            "lat": r[2] if r[2] else 17.38,
            "lon": r[3] if r[3] else 78.48
        })

    return jsonify(data)


# ================= ROUTE TYPE =================
@app.route('/get_route_type')
def get_route_type():

    if 'user_id' not in session:
        return jsonify({"type":"normal"})

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
    SELECT AVG(score) FROM driving_data
    WHERE user_id=%s
    """, (session['user_id'],))

    avg_score = cur.fetchone()[0]
    conn.close()

    if avg_score is None:
        return jsonify({"type":"normal"})

    if avg_score > 80:
        return jsonify({"type":"safe"})
    elif avg_score > 50:
        return jsonify({"type":"normal"})
    else:
        return jsonify({"type":"risky"})


# ================= ADMIN =================
@app.route('/admin')
def admin():
    if 'user_id' not in session or session['user_id'] != 999:
        return "Access Denied"

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id,username FROM users")
    users = cur.fetchall()

    cur.execute("""
    SELECT users.username,driving_data.deviations,driving_data.stops,
           driving_data.confusion,driving_data.score,driving_data.driver_type,
           driving_data.timestamp
    FROM driving_data
    JOIN users ON users.id = driving_data.user_id
    ORDER BY driving_data.timestamp DESC
    """)

    data = cur.fetchall()
    conn.close()

    return render_template('admin.html', users=users, data=data)


# ================= USER DETAIL =================
@app.route('/user/<username>')
def user_detail(username):

    if 'user_id' not in session or session['user_id'] != 999:
        return "Access Denied"

    conn = get_db()
    cur = conn.cursor()

    cur.execute("SELECT id FROM users WHERE username=%s", (username,))
    user = cur.fetchone()

    if not user:
        return "User not found"

    user_id = user[0]

    cur.execute("""
    SELECT deviations, stops, confusion, score, driver_type, timestamp
    FROM driving_data
    WHERE user_id=%s
    ORDER BY timestamp DESC
    """, (user_id,))

    data = cur.fetchall()
    conn.close()

    return render_template("user_detail.html", username=username, data=data)


# ================= RUN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)