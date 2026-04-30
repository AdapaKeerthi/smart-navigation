from flask import Flask, render_template, request, jsonify, redirect, session
import os
import psycopg2
from model import (model, predict_driver, predict_turn_risk,
                   predict_next_ride, behavior_forecast,
                   get_model_evaluation, retrain_from_db,
                   get_feature_importance, engineer_features)
from dotenv import load_dotenv
import bcrypt
from datetime import timedelta
import json
 
load_dotenv()
 
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secret_key_123")
# Detect environment
IS_PRODUCTION = os.environ.get("RENDER") or os.environ.get("PRODUCTION")
 
app.config.update(
    SESSION_COOKIE_SECURE=False,          # Must be False for localhost http://
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",        # Lax works for localhost
    SESSION_COOKIE_NAME="driveiq_session",
    PERMANENT_SESSION_LIFETIME=timedelta(days=7)
)
 
# ===================== DB =====================
def get_db():
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        return None
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    try:
        return psycopg2.connect(DATABASE_URL, sslmode='require')
    except Exception as e:
        print("DB ERROR:", e)
        return None
 
 
def init_db():
    conn = get_db()
    if not conn:
        return
    c = conn.cursor()
 
    # Core tables
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        id SERIAL PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        is_admin BOOLEAN DEFAULT FALSE
    )""")
 
    c.execute("""
    CREATE TABLE IF NOT EXISTS driving_data(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        deviations INTEGER DEFAULT 0,
        stops INTEGER DEFAULT 0,
        confusion INTEGER DEFAULT 0,
        score INTEGER DEFAULT 0,
        driver_type TEXT,
        latitude DOUBLE PRECISION DEFAULT 0,
        longitude DOUBLE PRECISION DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # NEW: confusion events (feature 2)
    c.execute("""
    CREATE TABLE IF NOT EXISTS confusion_events(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        event_type TEXT,        -- 'late_turn','sudden_stop','repeated_reroute','missed_turn'
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        junction_complexity INTEGER DEFAULT 1,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # NEW: route memory (feature 4, 30)
    c.execute("""
    CREATE TABLE IF NOT EXISTS route_memory(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        start_lat DOUBLE PRECISION,
        start_lon DOUBLE PRECISION,
        end_lat DOUBLE PRECISION,
        end_lon DOUBLE PRECISION,
        route_type TEXT,        -- 'fastest','easiest','balanced'
        mistakes INTEGER DEFAULT 0,
        confusion_count INTEGER DEFAULT 0,
        completed BOOLEAN DEFAULT TRUE,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # NEW: ride summaries (feature 24)
    c.execute("""
    CREATE TABLE IF NOT EXISTS ride_summaries(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        total_deviations INTEGER DEFAULT 0,
        total_stops INTEGER DEFAULT 0,
        total_confusion INTEGER DEFAULT 0,
        score INTEGER DEFAULT 0,
        driver_type TEXT,
        focus_level TEXT,
        route_type_used TEXT,
        duration_minutes INTEGER DEFAULT 0,
        distance_km DOUBLE PRECISION DEFAULT 0,
        suggestions TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # PHASE 2: danger zones table
    c.execute("""
    CREATE TABLE IF NOT EXISTS danger_zones(
        id SERIAL PRIMARY KEY,
        latitude DOUBLE PRECISION,
        longitude DOUBLE PRECISION,
        severity TEXT DEFAULT 'medium',
        description TEXT,
        source TEXT DEFAULT 'user_data',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # NEW: driver_features table for ML feature storage
    c.execute("""
    CREATE TABLE IF NOT EXISTS driver_features(
        id SERIAL PRIMARY KEY,
        user_id INTEGER,
        deviations INTEGER DEFAULT 0,
        stops INTEGER DEFAULT 0,
        confusion INTEGER DEFAULT 0,
        distance_km DOUBLE PRECISION DEFAULT 5,
        duration_minutes INTEGER DEFAULT 15,
        avg_speed DOUBLE PRECISION DEFAULT 30,
        turn_count INTEGER DEFAULT 5,
        driver_type TEXT,
        confidence DOUBLE PRECISION DEFAULT 0,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # NEW: model_evaluations table for tracking accuracy over time
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_evaluations(
        id SERIAL PRIMARY KEY,
        model_name TEXT,
        accuracy DOUBLE PRECISION,
        precision_score DOUBLE PRECISION,
        recall DOUBLE PRECISION,
        f1 DOUBLE PRECISION,
        dataset_size INTEGER,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
 
    # Migrations — safely add missing columns to existing tables
    for col_sql in [
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS is_admin BOOLEAN DEFAULT FALSE",
        "ALTER TABLE driving_data ADD COLUMN IF NOT EXISTS session_id TEXT",
        "ALTER TABLE driving_data ADD COLUMN IF NOT EXISTS latitude DOUBLE PRECISION DEFAULT 0",
        "ALTER TABLE driving_data ADD COLUMN IF NOT EXISTS longitude DOUBLE PRECISION DEFAULT 0",
    ]:
        try:
            c.execute(col_sql)
        except:
            pass
 
    conn.commit()
    conn.close()
 
init_db()
 
# ===================== HELPERS =====================
def is_admin():
    if 'user_id' not in session:
        print("is_admin: no user_id in session")
        return False
    conn = get_db()
    if not conn:
        print("is_admin: DB connection failed")
        return False
    cur = conn.cursor()
    cur.execute("SELECT is_admin FROM users WHERE id=%s", (session['user_id'],))
    row = cur.fetchone()
    conn.close()
    result = bool(row and row[0])
    print(f"is_admin check for user_id={session['user_id']}: {result}")
    return result
 
 
def get_driver_profile_data(user_id):
    """
    Returns computed driver profile dict from past rides.
    Used by multiple endpoints.
    """
    conn = get_db()
    if not conn:
        return None
    cur = conn.cursor()
 
    # Last 20 rides
    cur.execute("""
        SELECT deviations, stops, confusion, score, driver_type
        FROM driving_data
        WHERE user_id=%s
        ORDER BY timestamp DESC
        LIMIT 20
    """, (user_id,))
    rows = cur.fetchall()
 
    # Confusion events breakdown
    cur.execute("""
        SELECT event_type, COUNT(*) as cnt
        FROM confusion_events
        WHERE user_id=%s
        GROUP BY event_type
    """, (user_id,))
    events = {r[0]: r[1] for r in cur.fetchall()}
 
    # Route memory — which routes went badly
    cur.execute("""
        SELECT route_type, AVG(mistakes), AVG(confusion_count), COUNT(*)
        FROM route_memory
        WHERE user_id=%s
        GROUP BY route_type
    """, (user_id,))
    route_perf = {r[0]: {'avg_mistakes': r[1], 'avg_confusion': r[2], 'count': r[3]}
                  for r in cur.fetchall()}
 
    conn.close()
 
    if not rows:
        return {
            "skill_level": "Beginner",
            "avg_score": 0,
            "avg_deviations": 0,
            "avg_stops": 0,
            "avg_confusion": 0,
            "total_rides": 0,
            "confusion_events": events,
            "route_performance": route_perf,
            "avoid_complex_turns": False,
            "prefer_simple_routes": False,
            "needs_early_warnings": False,
            "voice_detail": "detailed",
            "weakness": "No data yet",
            "focus_score": 100,
        }
 
    n = len(rows)
    avg_dev  = sum(r[0] for r in rows) / n
    avg_stop = sum(r[1] for r in rows) / n
    avg_conf = sum(r[2] for r in rows) / n
    avg_score = sum(r[3] for r in rows) / n
 
    # Skill level
    if avg_score >= 75:
        skill = "Expert"
        voice_detail = "brief"
    elif avg_score >= 50:
        skill = "Average"
        voice_detail = "normal"
    else:
        skill = "Beginner"
        voice_detail = "detailed"
 
    # Behavior flags
    avoid_complex = avg_conf > 2 or events.get('late_turn', 0) > 3
    prefer_simple = avg_dev > 3 or events.get('missed_turn', 0) > 3
    needs_early   = avg_conf > 1.5 or events.get('repeated_reroute', 0) > 2
 
    # Weakness string
    weaknesses = []
    if avg_dev > 3:   weaknesses.append("frequent route deviations")
    if avg_conf > 2:  weaknesses.append("confusion at intersections")
    if avg_stop > 4:  weaknesses.append("unnecessary stops")
    if events.get('late_turn', 0) > 2: weaknesses.append("late turns")
    weakness_str = ", ".join(weaknesses) if weaknesses else "None detected"
 
    # Focus score (0-100)
    focus = max(0, 100 - (avg_dev * 4 + avg_conf * 6 + avg_stop * 2))
 
    return {
        "skill_level": skill,
        "avg_score": round(avg_score, 1),
        "avg_deviations": round(avg_dev, 1),
        "avg_stops": round(avg_stop, 1),
        "avg_confusion": round(avg_conf, 1),
        "total_rides": n,
        "confusion_events": events,
        "route_performance": route_perf,
        "avoid_complex_turns": avoid_complex,
        "prefer_simple_routes": prefer_simple,
        "needs_early_warnings": needs_early,
        "voice_detail": voice_detail,
        "weakness": weakness_str,
        "focus_score": round(focus, 1),
    }
 
 
# ===================== AUTH ROUTES =====================
@app.route('/')
def home():
    return redirect('/login')
 
 
@app.route('/map')
def map_page():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('index.html')
 
 
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db()
        if not conn:
            return render_template('login.html', error="Database error.")
        cur = conn.cursor()
        cur.execute("SELECT id, password FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        conn.close()
        if user:
            try:
                match = bcrypt.checkpw(password.encode(), user[1].encode())
            except:
                match = (user[1] == password)
            if match:
                session['user_id'] = user[0]
                session['username'] = username
                session.permanent = True
                return redirect('/map')
        return render_template('login.html', error="Invalid username or password.")
    return render_template('login.html')
 
 
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        if len(u) < 3 or len(p) < 6:
            return render_template('register.html', error="Username min 3 chars, password min 6 chars.")
        conn = get_db()
        if not conn:
            return render_template('register.html', error="Database error.")
        try:
            c = conn.cursor()
            c.execute("SELECT id FROM users WHERE username=%s", (u,))
            if c.fetchone():
                return render_template('register.html', error="Username already taken.")
            hashed = bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()
            c.execute("INSERT INTO users(username,password) VALUES(%s,%s)", (u, hashed))
            conn.commit()
            conn.close()
            return redirect('/login')
        except Exception as e:
            print("REGISTER ERROR:", e)
            return render_template('register.html', error="Registration failed.")
    return render_template('register.html')
 
 
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')
 
 
# ===================== DRIVER PROFILE API =====================
@app.route('/driver_profile')
def driver_profile():
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    profile = get_driver_profile_data(session['user_id'])
    return jsonify(profile)
 
 
# ===================== ROUTE SUGGESTION API (Feature 1, 25) =====================
@app.route('/suggest_routes', methods=['POST'])
def suggest_routes():
    """
    Receives start/end coords + driver profile.
    Returns 3 ranked routes: fastest, easiest, balanced.
    Each route scored for this specific driver.
    """
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
 
    data = request.json
    start = data.get('start')   # [lon, lat]
    end   = data.get('end')     # [lon, lat]
 
    if not start or not end:
        return jsonify({"error": "missing coords"}), 400
 
    profile = get_driver_profile_data(session['user_id'])
 
    # Build ORS route preferences
    route_configs = [
        {
            "id": "fastest",
            "label": "Fastest Route",
            "icon": "⚡",
            "ors_preference": "fastest",
            "ors_options": {}
        },
        {
            "id": "easiest",
            "label": "Easiest Route",
            "icon": "🟢",
            "ors_preference": "recommended",
            "ors_options": {
                "avoid_features": ["highways", "tollways"]
            }
        },
        {
            "id": "balanced",
            "label": "Balanced Route",
            "icon": "⚖️",
            "ors_preference": "recommended",
            "ors_options": {}
        }
    ]
 
    results = []
    for cfg in route_configs:
        body = {
            "coordinates": [start, end],
            "preference": cfg["ors_preference"],
            "instructions": True,
            "instructions_format": "text"
        }
        if cfg["ors_options"]:
            body["options"] = cfg["ors_options"]
 
        results.append({
            "id": cfg["id"],
            "label": cfg["label"],
            "icon": cfg["icon"],
            "body": body,
            "suitability": _score_route_for_driver(cfg["id"], profile),
            "reason": _explain_route(cfg["id"], profile)
        })
 
    # Sort: put the best-suited route first
    results.sort(key=lambda x: x["suitability"]["score"], reverse=True)
    results[0]["recommended"] = True
 
    # Save route memory start
    return jsonify({
        "routes": results,
        "profile": profile
    })
 
 
def _score_route_for_driver(route_id, profile):
    """Score a route type 0-100 for this driver profile."""
    skill = profile["skill_level"]
    avoid_complex = profile["avoid_complex_turns"]
    prefer_simple = profile["prefer_simple_routes"]
 
    scores = {
        "fastest": {"Expert": 95, "Average": 65, "Beginner": 40},
        "easiest": {"Expert": 60, "Average": 80, "Beginner": 95},
        "balanced": {"Expert": 75, "Average": 85, "Beginner": 70},
    }
 
    base = scores.get(route_id, {}).get(skill, 70)
 
    # Adjust for behavior flags
    if route_id == "fastest" and avoid_complex: base -= 20
    if route_id == "fastest" and prefer_simple: base -= 15
    if route_id == "easiest" and avoid_complex: base += 10
    if route_id == "easiest" and prefer_simple: base += 10
 
    base = max(0, min(100, base))
 
    label = "Best for you" if base >= 80 else ("Good match" if base >= 60 else "Not recommended")
    return {"score": base, "label": label}
 
 
def _explain_route(route_id, profile):
    """Explainable AI — say WHY this route suits or doesn't suit the driver."""
    skill = profile["skill_level"]
    avoid_complex = profile["avoid_complex_turns"]
    prefer_simple = profile["prefer_simple_routes"]
 
    if route_id == "fastest":
        if skill == "Expert":
            return "You're an experienced driver — this route suits you."
        elif avoid_complex:
            return "You often get confused at complex turns — this route has many."
        elif prefer_simple:
            return "You tend to deviate — this route has sharp turns."
        else:
            return "Standard fastest path."
 
    if route_id == "easiest":
        if prefer_simple:
            return "Chosen because you often deviate on complex roads."
        elif avoid_complex:
            return "Avoids complex intersections where you struggle."
        elif skill == "Expert":
            return "May feel slow for your skill level."
        else:
            return "Fewer turns, simpler junctions."
 
    if route_id == "balanced":
        return "Good mix of speed and simplicity for your driving style."
 
    return ""
 
 
# ===================== CONFUSION EVENT API (Feature 2) =====================
@app.route('/confusion_event', methods=['POST'])
def confusion_event():
    """
    Frontend sends real-time confusion signals.
    We store them and return adaptive guidance.
    """
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
 
    data = request.json
    event_type = data.get('event_type', 'unknown')
    lat = float(data.get('lat', 0))
    lon = float(data.get('lon', 0))
    junction_complexity = int(data.get('junction_complexity', 1))
 
    conn = get_db()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO confusion_events(user_id, event_type, latitude, longitude, junction_complexity)
            VALUES(%s, %s, %s, %s, %s)
        """, (session['user_id'], event_type, lat, lon, junction_complexity))
        conn.commit()
        conn.close()
 
    # Build adaptive response
    profile = get_driver_profile_data(session['user_id'])
    response = _get_adaptive_guidance(event_type, profile)
 
    return jsonify(response)
 
 
def _get_adaptive_guidance(event_type, profile):
    """Return voice message + action based on confusion event + driver profile."""
    skill = profile["skill_level"]
    needs_early = profile["needs_early_warnings"]
 
    guidance = {
        "action": "none",
        "voice_message": "",
        "show_arrows": False,
        "reroute": False,
        "explainer": ""
    }
 
    if event_type == "late_turn":
        if skill == "Beginner":
            guidance["voice_message"] = "Don't worry, I'll guide you again. Rerouting now."
            guidance["show_arrows"] = True
        else:
            guidance["voice_message"] = "Missed turn. Recalculating."
        guidance["reroute"] = True
        guidance["explainer"] = "Rerouting because you missed the turn."
 
    elif event_type == "sudden_stop":
        if skill == "Beginner":
            guidance["voice_message"] = "Take your time. The junction ahead is complex. Turn right in 80 meters."
            guidance["show_arrows"] = True
        else:
            guidance["voice_message"] = "Complex junction ahead. Prepare to turn."
        guidance["action"] = "slow_warning"
 
    elif event_type == "repeated_reroute":
        guidance["voice_message"] = "You've rerouted multiple times. Switching to an easier path."
        guidance["reroute"] = True
        guidance["action"] = "switch_to_easy"
        guidance["explainer"] = "Switching to easier route because you've rerouted several times."
 
    elif event_type == "approaching_complex":
        if needs_early or skill == "Beginner":
            guidance["voice_message"] = "Complex intersection in 200 meters. Prepare early."
            guidance["show_arrows"] = True
            guidance["action"] = "early_warning"
        elif skill == "Average":
            guidance["voice_message"] = "Complex junction ahead. Stay alert."
 
    elif event_type == "predictive_miss":
        guidance["voice_message"] = "You might miss this turn — it's coming up on your right."
        guidance["show_arrows"] = True
        guidance["action"] = "predictive_warning"
 
    return guidance
 
 
# ===================== SAVE BEHAVIOR (Feature 5 enhanced with full ML) =====================
@app.route('/save_behavior', methods=['POST'])
def save_behavior():
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    try:
        data = request.json
        d   = int(data.get('deviations', 0))
        s   = int(data.get('stops', 0))
        c   = int(data.get('confusion', 0))
        lat = float(data.get('lat', 0))
        lon = float(data.get('lon', 0))
        distance_km      = float(data.get('distance_km', 5.0))
        duration_minutes = int(data.get('duration_minutes', 15))
        avg_speed        = float(data.get('avg_speed', 30.0))
        turn_count       = int(data.get('turn_count', 5))
 
        score = max(0, 100 - (d * 5 + s * 3 + c * 4))
 
        # Full ML prediction with feature engineering
        try:
            ml_result   = predict_driver(d, s, c, distance_km, duration_minutes,
                                         avg_speed, turn_count)
            driver_type = ml_result["driver_type"]
            confidence  = ml_result["confidence"]
        except Exception as ex:
            print("ML error:", ex)
            driver_type, confidence = "Unknown", 0.0
 
        focus_score = max(0, 100 - (d * 4 + c * 6 + s * 2))
        focus_level = "High" if focus_score >= 75 else ("Medium" if focus_score >= 50 else "Low")
 
        conn = get_db()
        if not conn:
            return jsonify({"error": "DB error"}), 500
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO driving_data(user_id,deviations,stops,confusion,score,driver_type,latitude,longitude)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s)
        """, (session['user_id'], d, s, c, score, driver_type, lat, lon))
 
        # Store engineered features for adaptive retraining
        cur.execute("""
            INSERT INTO driver_features(user_id,deviations,stops,confusion,
                distance_km,duration_minutes,avg_speed,turn_count,driver_type,confidence)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (session['user_id'], d, s, c, distance_km, duration_minutes,
              avg_speed, turn_count, driver_type, confidence))
        conn.commit()
        conn.close()
 
        return jsonify({
            "score": score,
            "driver_type": driver_type,
            "confidence": confidence,
            "focus_level": focus_level,
            "focus_score": focus_score,
            "ml_breakdown": ml_result.get("breakdown", {}),
        })
    except Exception as e:
        print("SAVE ERROR:", e)
        return jsonify({"error": "save failed"}), 500
 
 
# ===================== RIDE SUMMARY API (Feature 24) =====================
@app.route('/ride_summary', methods=['POST'])
def ride_summary():
    """Called when navigation ends. Saves and returns full ride summary."""
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
 
    data = request.json
    d         = int(data.get('deviations', 0))
    s         = int(data.get('stops', 0))
    c         = int(data.get('confusion', 0))
    duration  = int(data.get('duration_minutes', 0))
    distance  = float(data.get('distance_km', 0))
    route_type = data.get('route_type', 'unknown')
 
    score = max(0, 100 - (d * 5 + s * 3 + c * 4))
    focus_score = max(0, 100 - (d * 4 + c * 6 + s * 2))
 
    try:
        driver_type = model.predict([[d, s, c]])[0]
    except:
        driver_type = "Unknown"
 
    focus_level = "High" if focus_score >= 75 else ("Medium" if focus_score >= 50 else "Low")
 
    # Generate suggestions
    suggestions = []
    if d > 3:  suggestions.append(f"You deviated {d} times — try following the route more closely.")
    if c > 2:  suggestions.append(f"You had {c} confusion events — enable early warnings next time.")
    if s > 5:  suggestions.append(f"You stopped {s} times — practice smoother driving.")
    if score >= 80: suggestions.append("Excellent drive! Keep it up.")
    elif score >= 60: suggestions.append("Good drive with some room to improve.")
    else: suggestions.append("Consider taking easier routes to build confidence.")
 
    suggestions_str = " | ".join(suggestions)
 
    conn = get_db()
    if conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO ride_summaries(
                user_id,total_deviations,total_stops,total_confusion,
                score,driver_type,focus_level,route_type_used,
                duration_minutes,distance_km,suggestions
            ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (session['user_id'], d, s, c, score, driver_type, focus_level,
              route_type, duration, distance, suggestions_str))
        conn.commit()
        conn.close()
 
    return jsonify({
        "score": score,
        "driver_type": driver_type,
        "focus_level": focus_level,
        "focus_score": focus_score,
        "deviations": d,
        "stops": s,
        "confusion": c,
        "duration_minutes": duration,
        "distance_km": distance,
        "suggestions": suggestions,
        "route_type": route_type
    })
 
 
# ===================== HEATMAP DATA API (Feature 20) =====================
@app.route('/heatmap_data')
def heatmap_data():
    if 'user_id' not in session:
        return jsonify([])
    conn = get_db()
    if not conn:
        return jsonify([])
    cur = conn.cursor()
    cur.execute("""
        SELECT latitude, longitude, COUNT(*) as intensity
        FROM confusion_events
        WHERE user_id=%s AND latitude != 0 AND longitude != 0
        GROUP BY latitude, longitude
    """, (session['user_id'],))
    rows = cur.fetchall()
    conn.close()
    return jsonify([{"lat": r[0], "lon": r[1], "intensity": r[2]} for r in rows])
 
 
# ===================== LIVE DATA (Admin) =====================
@app.route('/live_data')
def live_data():
    if not is_admin():
        return jsonify({"error": "Access denied"}), 403
    conn = get_db()
    if not conn:
        return jsonify([])
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT ON (u.id)
            u.username, d.score, d.driver_type, d.latitude, d.longitude, d.timestamp
        FROM users u
        JOIN driving_data d ON u.id = d.user_id
        WHERE d.timestamp > NOW() - INTERVAL '10 minutes'
          AND d.latitude != 0 AND d.longitude != 0
        ORDER BY u.id, d.timestamp DESC
    """)
    rows = cur.fetchall()
    conn.close()
    return jsonify([{
        "username": r[0], "score": r[1], "driver_type": r[2],
        "lat": r[3], "lon": r[4], "timestamp": str(r[5])
    } for r in rows])
 
 
# ===================== DASHBOARD =====================
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect('/login')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT deviations, stops, confusion, score, driver_type, timestamp
        FROM driving_data WHERE user_id=%s ORDER BY timestamp DESC LIMIT 50
    """, (session['user_id'],))
    data = cur.fetchall()
 
    # Ride summaries for improvement graph
    cur.execute("""
        SELECT score, focus_level, timestamp, suggestions
        FROM ride_summaries WHERE user_id=%s ORDER BY timestamp DESC LIMIT 20
    """, (session['user_id'],))
    summaries = cur.fetchall()
 
    conn.close()
    profile = get_driver_profile_data(session['user_id'])
    username = session.get('username', 'Driver')
    return render_template('dashboard.html', data=data, username=username,
                           profile=profile, summaries=summaries)
 
 
# ===================== ADMIN =====================
@app.route('/admin')
def admin():
    if not is_admin():
        return redirect('/login')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users ORDER BY id")
    users = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM driving_data")
    total_trips = cur.fetchone()[0]
    cur.execute("SELECT AVG(score) FROM driving_data")
    avg_score_row = cur.fetchone()[0]
    avg_score = round(avg_score_row, 1) if avg_score_row else 0
    conn.close()
    return render_template('admin.html', users=users, total_trips=total_trips, avg_score=avg_score)
 
 
@app.route('/user/<username>')
def user_detail(username):
    if not is_admin():
        return redirect('/login')
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username=%s", (username,))
    user = cur.fetchone()
    if not user:
        return "User not found", 404
    cur.execute("""
        SELECT deviations, stops, confusion, score, driver_type, timestamp
        FROM driving_data WHERE user_id=%s ORDER BY timestamp DESC
    """, (user[0],))
    data = cur.fetchall()
    conn.close()
    return render_template("user_detail.html", username=username, data=data)
 
 
@app.route('/make_admin/<username>/<secret>')
def make_admin(username, secret):
    if secret != os.environ.get("ADMIN_SECRET", "changeme123"):
        return "Invalid", 403
    conn = get_db()
    cur = conn.cursor()
    cur.execute("UPDATE users SET is_admin=TRUE WHERE username=%s", (username,))
    conn.commit()
    conn.close()
    return f"{username} is now admin ✅"
 
 
 
 
# ===================== PHASE 2: HEATMAP PAGE =====================
@app.route('/heatmap')
def heatmap_page():
    if 'user_id' not in session:
        return redirect('/login')
    return render_template('heatmap.html')
 
 
# ===================== PHASE 2: DANGER ZONES =====================
@app.route('/danger_zones')
def danger_zones():
    """
    Returns danger zones = locations with 3+ confusion events.
    Used by frontend to warn driver when approaching.
    """
    if 'user_id' not in session:
        return jsonify([])
    conn = get_db()
    if not conn:
        return jsonify([])
    cur = conn.cursor()
 
    # Personal danger zones (user's own confusion hotspots)
    cur.execute("""
        SELECT latitude, longitude, COUNT(*) as hits, array_agg(event_type) as types
        FROM confusion_events
        WHERE latitude != 0 AND longitude != 0
        GROUP BY ROUND(latitude::numeric,3), ROUND(longitude::numeric,3), latitude, longitude
        HAVING COUNT(*) >= 2
        ORDER BY hits DESC
        LIMIT 50
    """)
    rows = cur.fetchall()
    conn.close()
 
    zones = []
    for r in rows:
        types = r[3] or []
        worst = 'repeated_reroute' if 'repeated_reroute' in types else                 'late_turn' if 'late_turn' in types else                 'sudden_stop' if 'sudden_stop' in types else 'confusion'
        zones.append({
            "lat": r[0], "lon": r[1],
            "hits": r[2],
            "severity": "high" if r[2] >= 5 else "medium" if r[2] >= 3 else "low",
            "type": worst
        })
    return jsonify(zones)
 
 
# ===================== PHASE 2: TURN DIFFICULTY =====================
@app.route('/turn_difficulty', methods=['POST'])
def turn_difficulty():
    """
    Predicts difficulty of upcoming turns based on:
    - Turn angle (from instruction type)
    - Past user confusion at similar turns
    - Junction complexity (number of roads)
    """
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
 
    data = request.json
    turns = data.get('turns', [])  # list of {instruction, lat, lon, distance}
 
    conn = get_db()
    cur = conn.cursor() if conn else None
 
    results = []
    for t in turns:
        instruction = t.get('instruction', '').lower()
        lat = t.get('lat', 0)
        lon = t.get('lon', 0)
 
        # Base difficulty from turn type
        if any(w in instruction for w in ['sharp right', 'sharp left', 'u-turn']):
            base = 3  # Hard
        elif any(w in instruction for w in ['right', 'left', 'roundabout']):
            base = 2  # Medium
        elif any(w in instruction for w in ['slight right', 'slight left']):
            base = 1  # Easy
        else:
            base = 1  # Straight/easy
 
        # Check if user has past confusion near this location
        past_confusion = 0
        if cur and lat and lon:
            cur.execute("""
                SELECT COUNT(*) FROM confusion_events
                WHERE user_id=%s
                  AND ABS(latitude - %s) < 0.005
                  AND ABS(longitude - %s) < 0.005
            """, (session['user_id'], lat, lon))
            row = cur.fetchone()
            past_confusion = row[0] if row else 0
 
        if past_confusion >= 3:
            base = min(3, base + 1)
        elif past_confusion >= 1:
            base = min(3, base + 0)
 
        difficulty = {1: 'Easy', 2: 'Medium', 3: 'Hard'}[base]
        emoji      = {1: '🟢',   2: '🟡',    3: '🔴'}[base]
 
        results.append({
            "instruction": t.get('instruction'),
            "difficulty": difficulty,
            "emoji": emoji,
            "past_confusion": past_confusion,
            "distance": t.get('distance', 0)
        })
 
    if conn:
        conn.close()
 
    return jsonify(results)
 
 
# ===================== PHASE 2: ROUTE MEMORY =====================
@app.route('/save_route_memory', methods=['POST'])
def save_route_memory():
    """Save completed route info for learning."""
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    data = request.json
    conn = get_db()
    if not conn:
        return jsonify({"ok": False})
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO route_memory(
            user_id, start_lat, start_lon, end_lat, end_lon,
            route_type, mistakes, confusion_count, completed
        ) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        session['user_id'],
        float(data.get('start_lat', 0)), float(data.get('start_lon', 0)),
        float(data.get('end_lat', 0)),   float(data.get('end_lon', 0)),
        data.get('route_type', 'unknown'),
        int(data.get('mistakes', 0)),
        int(data.get('confusion', 0)),
        bool(data.get('completed', True))
    ))
    conn.commit()
    conn.close()
    return jsonify({"ok": True})
 
 
@app.route('/route_memory_lookup', methods=['POST'])
def route_memory_lookup():
    """
    Check if user has done this route before.
    Returns best route type + what went wrong last time.
    """
    if 'user_id' not in session:
        return jsonify({"known": False})
    data  = request.json
    slat  = float(data.get('start_lat', 0))
    slon  = float(data.get('start_lon', 0))
    elat  = float(data.get('end_lat', 0))
    elon  = float(data.get('end_lon', 0))
 
    conn = get_db()
    if not conn:
        return jsonify({"known": False})
    cur = conn.cursor()
 
    # Find similar routes (within ~500m of start and end)
    cur.execute("""
        SELECT route_type, mistakes, confusion_count, completed, timestamp
        FROM route_memory
        WHERE user_id=%s
          AND ABS(start_lat-%s) < 0.005 AND ABS(start_lon-%s) < 0.005
          AND ABS(end_lat-%s)   < 0.005 AND ABS(end_lon-%s)   < 0.005
        ORDER BY timestamp DESC
        LIMIT 5
    """, (session['user_id'], slat, slon, elat, elon))
 
    rows = cur.fetchall()
    conn.close()
 
    if not rows:
        return jsonify({"known": False})
 
    # Summarize past performance
    best_route = min(rows, key=lambda r: r[1] + r[2])  # fewest mistakes+confusion
    avg_mistakes = sum(r[1] for r in rows) / len(rows)
    avg_confusion = sum(r[2] for r in rows) / len(rows)
 
    tips = []
    if avg_mistakes > 2:
        tips.append(f"You struggled on this route before ({avg_mistakes:.0f} deviations avg). Take it slow.")
    if avg_confusion > 1:
        tips.append("You had confusion events here before. Early warnings enabled.")
    if best_route[3]:
        tips.append(f"Your best attempt used the {best_route[0]} route.")
 
    return jsonify({
        "known": True,
        "times_done": len(rows),
        "best_route_type": best_route[0],
        "avg_mistakes": round(avg_mistakes, 1),
        "avg_confusion": round(avg_confusion, 1),
        "tips": tips
    })
 
 
# ===================== PHASE 2: GLOBAL HEATMAP (admin) =====================
@app.route('/global_heatmap')
def global_heatmap():
    """All users confusion events — for admin heatmap."""
    if not is_admin():
        return jsonify([])
    conn = get_db()
    if not conn:
        return jsonify([])
    cur = conn.cursor()
    cur.execute("""
        SELECT latitude, longitude, COUNT(*) as intensity
        FROM confusion_events
        WHERE latitude != 0 AND longitude != 0
        GROUP BY ROUND(latitude::numeric,3), ROUND(longitude::numeric,3), latitude, longitude
        ORDER BY intensity DESC
        LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()
    return jsonify([{"lat": r[0], "lon": r[1], "intensity": r[2]} for r in rows])
 
# ══════════════════════════════════════════════════════════════════════════════
#  ██████  ML / AI INTELLIGENCE ENDPOINTS (8 new features)
# ══════════════════════════════════════════════════════════════════════════════
 
# ── 1. FULL ML PREDICTION (Logistic Regression + RF + Neural Net ensemble) ──
@app.route('/ml_predict', methods=['POST'])
def ml_predict():
    """
    Rich driver classification using all three ML models.
    Accepts full ride context for feature engineering.
    """
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    data = request.json or {}
    try:
        result = predict_driver(
            deviations       = int(data.get('deviations', 0)),
            stops            = int(data.get('stops', 0)),
            confusion        = int(data.get('confusion', 0)),
            distance_km      = float(data.get('distance_km', 5.0)),
            duration_minutes = int(data.get('duration_minutes', 15)),
            avg_speed        = float(data.get('avg_speed', 30.0)),
            turn_count       = int(data.get('turn_count', 5)),
        )
        # Attach engineered features for transparency
        result["features_used"] = engineer_features(
            int(data.get('deviations', 0)), int(data.get('stops', 0)),
            int(data.get('confusion', 0)), float(data.get('distance_km', 5.0)),
            int(data.get('duration_minutes', 15)), float(data.get('avg_speed', 30.0)),
            int(data.get('turn_count', 5))
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
# ── 2. PREDICTIVE TURN RISK (predict miss BEFORE it happens) ──────────────────
@app.route('/predict_turn', methods=['POST'])
def predict_turn():
    """
    Predict probability the driver will miss the upcoming turn.
    Called by frontend when approaching a turn.
    """
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    data = request.json or {}
    profile = get_driver_profile_data(session['user_id'])
    turn_instruction    = data.get('instruction', 'turn right')
    distance_to_turn    = float(data.get('distance', 100))
    lat = float(data.get('lat', 0))
    lon = float(data.get('lon', 0))
 
    # Count historical confusion events near this location
    past_confusion = 0
    conn = get_db()
    if conn and lat and lon:
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM confusion_events
            WHERE user_id=%s AND ABS(latitude-%s)<0.005 AND ABS(longitude-%s)<0.005
        """, (session['user_id'], lat, lon))
        row = cur.fetchone()
        past_confusion = row[0] if row else 0
        conn.close()
 
    result = predict_turn_risk(profile, turn_instruction, distance_to_turn, past_confusion)
    result["past_confusion_nearby"] = past_confusion
    return jsonify(result)
 
 
# ── 3. NEXT RIDE FORECAST (behavior prediction) ───────────────────────────────
@app.route('/predict_next_ride')
def predict_next_ride_route():
    """
    Forecast the user's next ride risk level based on recent history.
    Provides trend analysis and predicted score.
    """
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    conn = get_db()
    if not conn:
        return jsonify({"error": "DB error"}), 500
    cur = conn.cursor()
    cur.execute("""
        SELECT deviations, stops, confusion, score, driver_type
        FROM driving_data WHERE user_id=%s ORDER BY timestamp DESC LIMIT 20
    """, (session['user_id'],))
    rows = cur.fetchall()
    conn.close()
 
    forecast = predict_next_ride(rows)
    forecast["score_forecast"] = behavior_forecast(rows, future_rides=5)
    return jsonify(forecast)
 
 
# ── 4. MODEL EVALUATION REPORT ────────────────────────────────────────────────
@app.route('/model_evaluation')
def model_evaluation():
    """
    Returns accuracy, precision, recall, F1 for all 3 trained models.
    Also returns feature importance from Random Forest.
    Admin only.
    """
    if not is_admin():
        return jsonify({"error": "admin only"}), 403
    report = get_model_evaluation()
    return jsonify(report)
 
 
# ── 5. ADAPTIVE RETRAINING ────────────────────────────────────────────────────
@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """
    Triggers model retraining using real DB data.
    Admin only. Loads labelled rides and retrains all 3 models.
    """
    if not is_admin():
        return jsonify({"error": "admin only"}), 403
    conn = get_db()
    if not conn:
        return jsonify({"error": "DB error"}), 500
    cur = conn.cursor()
    cur.execute("""
        SELECT deviations, stops, confusion, score, driver_type
        FROM driving_data WHERE driver_type IS NOT NULL AND driver_type != 'Unknown'
        ORDER BY timestamp DESC LIMIT 500
    """)
    rows = cur.fetchall()
 
    result = retrain_from_db(rows)
 
    # Log evaluation to DB
    if result.get("retrained") and result.get("metrics"):
        for model_name, m in result["metrics"].items():
            cur.execute("""
                INSERT INTO model_evaluations(model_name,accuracy,precision_score,recall,f1,dataset_size)
                VALUES(%s,%s,%s,%s,%s,%s)
            """, (model_name, m["accuracy"], m["precision"], m["recall"], m["f1"], result["rows_used"]))
        conn.commit()
    conn.close()
    return jsonify(result)
 
 
# ── 6. FEATURE IMPORTANCE ─────────────────────────────────────────────────────
@app.route('/feature_importance')
def feature_importance_route():
    """Returns which driving features matter most for classification."""
    if 'user_id' not in session:
        return jsonify({"error": "login required"}), 401
    return jsonify(get_feature_importance())
 
 
# ── 7. USER CLASSIFICATION ANALYTICS ─────────────────────────────────────────
@app.route('/analytics')
def analytics():
    """
    Full analytics for the current user:
    - Driver classification history
    - Score progression
    - Behavior feature trends
    - Next ride prediction
    - Model confidence over time
    """
    if 'user_id' not in session:
        return redirect('/login')
    conn = get_db()
    if not conn:
        return render_template('analytics.html', error="DB error")
    cur = conn.cursor()
 
    # Full ride history
    cur.execute("""
        SELECT deviations, stops, confusion, score, driver_type, timestamp
        FROM driving_data WHERE user_id=%s ORDER BY timestamp DESC LIMIT 50
    """, (session['user_id'],))
    rides = cur.fetchall()
 
    # Feature history
    cur.execute("""
        SELECT deviations, stops, confusion, distance_km, duration_minutes,
               avg_speed, turn_count, driver_type, confidence, timestamp
        FROM driver_features WHERE user_id=%s ORDER BY timestamp DESC LIMIT 50
    """, (session['user_id'],))
    features = cur.fetchall()
 
    # Driver type distribution
    cur.execute("""
        SELECT driver_type, COUNT(*) as cnt
        FROM driving_data WHERE user_id=%s AND driver_type IS NOT NULL
        GROUP BY driver_type ORDER BY cnt DESC
    """, (session['user_id'],))
    type_dist = cur.fetchall()
 
    # Model evaluation history (global)
    cur.execute("""
        SELECT model_name, accuracy, f1, evaluated_at
        FROM model_evaluations ORDER BY evaluated_at DESC LIMIT 20
    """)
    eval_history = cur.fetchall()
 
    conn.close()
 
    profile  = get_driver_profile_data(session['user_id'])
    forecast = predict_next_ride(rides) if rides else {}
    score_fc = behavior_forecast(rides, 5) if rides else []
    fi       = get_feature_importance()
    username = session.get('username', 'Driver')
 
    return render_template('analytics.html',
        username=username, rides=rides, features=features,
        type_dist=type_dist, eval_history=eval_history,
        profile=profile, forecast=forecast, score_forecast=score_fc,
        feature_importance=fi)
 
 
# ── 8. GLOBAL DRIVER CLASSIFICATION (Admin) ───────────────────────────────────
@app.route('/admin_classifications')
def admin_classifications():
    """
    Admin view: classify all users, show distribution, model performance.
    """
    if not is_admin():
        return jsonify({"error": "admin only"}), 403
    conn = get_db()
    if not conn:
        return jsonify([])
    cur = conn.cursor()
    cur.execute("""
        SELECT u.username,
               COUNT(d.id) as trips,
               AVG(d.score) as avg_score,
               AVG(d.deviations) as avg_dev,
               AVG(d.confusion) as avg_conf,
               MODE() WITHIN GROUP (ORDER BY d.driver_type) as common_type
        FROM users u
        LEFT JOIN driving_data d ON u.id = d.user_id
        GROUP BY u.username
        ORDER BY avg_score DESC NULLS LAST
    """)
    rows = cur.fetchall()
 
    # Model accuracy history
    cur.execute("""
        SELECT model_name, AVG(accuracy) as avg_acc, COUNT(*) as evals
        FROM model_evaluations GROUP BY model_name
    """)
    model_stats = cur.fetchall()
 
    conn.close()
 
    users = []
    for r in rows:
        username, trips, avg_score, avg_dev, avg_conf, common_type = r
        avg_score = round(float(avg_score), 1) if avg_score else 0
        avg_dev   = round(float(avg_dev), 1) if avg_dev else 0
        avg_conf  = round(float(avg_conf), 1) if avg_conf else 0
 
        # Re-classify using ML
        try:
            ml = predict_driver(int(avg_dev), 0, int(avg_conf))
            ml_type = ml["driver_type"]
            ml_conf = ml["confidence"]
        except Exception:
            ml_type, ml_conf = common_type or "Unknown", 0
 
        users.append({
            "username": username, "trips": trips or 0,
            "avg_score": avg_score, "avg_dev": avg_dev,
            "avg_conf": avg_conf, "stored_type": common_type,
            "ml_type": ml_type, "ml_confidence": ml_conf
        })
 
    return jsonify({
        "users": users,
        "model_stats": [{"model": r[0], "avg_accuracy": round(float(r[1]),4),
                          "evaluations": r[2]} for r in model_stats]
    })
 
 
# ===================== RUN =====================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)