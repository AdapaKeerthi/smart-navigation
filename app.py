from flask import Flask, render_template, request, jsonify, redirect, session
import os
import psycopg2
from model import model   # AI model

app = Flask(__name__)
app.secret_key = "secret123"

# ================= DATABASE (POSTGRESQL) =================
DATABASE_URL = os.environ.get("DATABASE_URL")

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
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

# Insert admin user (only once)
cursor.execute("""
INSERT INTO users (id, username, password)
VALUES (999, 'admin', 'admin123')
ON CONFLICT (id) DO NOTHING
""")
# ========================================================


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

        try:
            cursor.execute(
                "INSERT INTO users (username,password) VALUES (%s,%s)",
                (u,p)
            )
        except:
            return "User already exists"

        return redirect('/login')

    return render_template('register.html')


# ================= LOGIN =================
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']

        cursor.execute(
            "SELECT * FROM users WHERE username=%s AND password=%s",
            (u,p)
        )
        user = cursor.fetchone()

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

    score = 100 - (deviations*5 + stops*3 + confusion*4)

    # AI Prediction
    prediction = model.predict([[deviations, stops, confusion]])
    driver_type = prediction[0]

    cursor.execute("""
    INSERT INTO driving_data 
    (user_id, deviations, stops, confusion, score, driver_type)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (session['user_id'], deviations, stops, confusion, score, driver_type))

    return jsonify({"score":score,"driver_type":driver_type})


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:
        return redirect('/login')

    cursor.execute(
        "SELECT * FROM driving_data WHERE user_id=%s ORDER BY timestamp DESC",
        (session['user_id'],)
    )
    rows = cursor.fetchall()

    return render_template('dashboard.html', data=rows)


# ================= ADMIN =================
@app.route('/admin')
def admin_panel():

    if 'user_id' not in session or session['user_id'] != 999:
        return "Access Denied"

    cursor.execute("SELECT id, username FROM users")
    users = cursor.fetchall()

    cursor.execute("""
        SELECT users.username, driving_data.deviations, driving_data.stops,
               driving_data.confusion, driving_data.score, driving_data.driver_type,
               driving_data.timestamp
        FROM driving_data
        JOIN users ON users.id = driving_data.user_id
        ORDER BY driving_data.timestamp DESC
    """)
    data = cursor.fetchall()

    return render_template('admin.html', users=users, data=data)


# ================= RUN =================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)