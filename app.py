from flask import Flask, render_template, request, jsonify, redirect, session
import os
import psycopg2
try:
    from model import model
except:
    model = None   # ✅ AI model

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE CONNECTION =================
def get_db():
    return psycopg2.connect(
        os.environ.get("DATABASE_URL"),
        sslmode='require'
    )

# ================= INIT TABLES =================
def init_db():
    conn = get_db()
    c = conn.cursor()

    # Users table
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT
    )
    """)

    # Driving data table
    c.execute("""
    CREATE TABLE IF NOT EXISTS driving_data (
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        deviations INTEGER,
        stops INTEGER,
        confusion INTEGER,
        score INTEGER,
        driver_type TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Admin user
    c.execute("""
    INSERT INTO users (id, username, password)
    VALUES (999, 'admin', 'admin123')
    ON CONFLICT (id) DO NOTHING
    """)

    conn.commit()
    conn.close()

init_db()
# =======================================================


@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('index.html')


# ================= REGISTER =================
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']

        conn = get_db()
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users (username,password) VALUES (%s,%s)", (u,p))
            conn.commit()
        except:
            conn.close()
            return "User already exists"

        conn.close()
        return redirect('/login')

    return render_template('register.html')


# ================= LOGIN =================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']

        conn = get_db()
        c = conn.cursor()

        c.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (u,p)
        )
        user = c.fetchone()

        conn.close()

        if user:
            session['user_id'] = user[0]
            return redirect('/')
        else:
            return "Invalid login"

    return render_template('login.html')


# ================= LOGOUT =================
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# ================= SAVE DATA =================
@app.route('/save_behavior', methods=['POST'])
def save_behavior():

    if 'user_id' not in session:
        return jsonify({"error":"not logged in"})

    data = request.json

    deviations = data.get('deviations', 0)
    stops = data.get('stops', 0)
    confusion = data.get('confusion', 0)

    # Score calculation
    score = 100 - (deviations*5 + stops*3 + confusion*4)
    if score < 0:
        score = 0

    # ✅ AI Prediction
if model:
    prediction = model.predict([[deviations, stops, confusion]])
    driver_type = prediction[0]
else:
    driver_type = "Normal"

    conn = get_db()
    c = conn.cursor()

    c.execute("""
    INSERT INTO driving_data 
    (user_id, deviations, stops, confusion, score, driver_type)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (session['user_id'], deviations, stops, confusion, score, driver_type))

    conn.commit()
    conn.close()

    return jsonify({"score":score,"driver_type":driver_type})


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:
        return redirect('/login')

    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT * FROM driving_data WHERE user_id=%s ORDER BY timestamp DESC",
        (session['user_id'],)
    )
    rows = c.fetchall()

    conn.close()

    return render_template('dashboard.html', data=rows)


# ================= ADMIN =================
@app.route('/admin')
def admin_panel():

    if 'user_id' not in session or session['user_id'] != 999:
        return "Access Denied"

    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT id, username FROM users")
    users = c.fetchall()

    c.execute("""
        SELECT users.username, driving_data.deviations, driving_data.stops,
               driving_data.confusion, driving_data.score, driving_data.driver_type,
               driving_data.timestamp
        FROM driving_data
        JOIN users ON users.id = driving_data.user_id
        ORDER BY driving_data.timestamp DESC
    """)
    data = c.fetchall()

    conn.close()

    return render_template('admin.html', users=users, data=data)


# ================= RUN =================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)