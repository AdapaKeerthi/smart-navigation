from flask import Flask, render_template, request, jsonify, redirect, session
import sqlite3

app = Flask(__name__)
app.secret_key = "secret123"  # needed for login session

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('drivers.db')
    c = conn.cursor()

    # users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
    ''')
    

    # driving data
    c.execute('''
    CREATE TABLE IF NOT EXISTS driving_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        deviations INTEGER,
        stops INTEGER,
        confusion INTEGER,
        score INTEGER,
        driver_type TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    c.execute("INSERT OR IGNORE INTO users (id, username, password) VALUES (999, 'admin', 'admin123')")

    conn.commit()
    conn.close()

init_db()
# ============================================


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

        conn = sqlite3.connect('drivers.db')
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users (username,password) VALUES (?,?)",(u,p))
            conn.commit()
        except:
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

        conn = sqlite3.connect('drivers.db')
        c = conn.cursor()

        user = c.execute(
            "SELECT * FROM users WHERE username=? AND password=?",
            (u,p)
        ).fetchone()

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

    score = 100 - (deviations*5 + stops*3 + confusion*4)

    if score > 80:
        driver_type = "Safe Driver"
    elif score > 50:
        driver_type = "Average Driver"
    else:
        driver_type = "Risky Driver"

    conn = sqlite3.connect('drivers.db')
    c = conn.cursor()

    c.execute("""
    INSERT INTO driving_data 
    (user_id, deviations, stops, confusion, score, driver_type)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (session['user_id'], deviations, stops, confusion, score, driver_type))

    conn.commit()
    conn.close()

    return jsonify({"score":score,"driver_type":driver_type})


# ================= DASHBOARD =================
@app.route('/dashboard')
def dashboard():

    if 'user_id' not in session:
        return redirect('/login')

    conn = sqlite3.connect('drivers.db')
    c = conn.cursor()

    rows = c.execute(
        "SELECT * FROM driving_data WHERE user_id=? ORDER BY timestamp DESC",
        (session['user_id'],)
    ).fetchall()

    conn.close()

    return render_template('dashboard.html', data=rows)


if __name__ == '__main__':
    app.run(debug=True)
@app.route('/admin')
def admin_panel():

    if 'user_id' not in session or session['user_id'] != 999:
        return "Access Denied"

    conn = sqlite3.connect('drivers.db')
    c = conn.cursor()

    users = c.execute("SELECT id, username FROM users").fetchall()

    data = c.execute("""
        SELECT users.username, driving_data.deviations, driving_data.stops,
               driving_data.confusion, driving_data.score, driving_data.driver_type,
               driving_data.timestamp
        FROM driving_data
        JOIN users ON users.id = driving_data.user_id
        ORDER BY driving_data.timestamp DESC
    """).fetchall()

    conn.close()

    return render_template('admin.html', users=users, data=data)