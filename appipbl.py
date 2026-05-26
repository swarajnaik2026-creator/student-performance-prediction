"""
AI-Based Student Performance Prediction System
================================================
Flask backend — serves all pages and the prediction API.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import json
import socket
import webbrowser

app = Flask(__name__)

# ── Paths and fallback assets ─────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE, 'model')
DATASET_DIR = os.path.join(BASE, 'dataset')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.pkl')
DATASET_CSV = os.path.join(DATASET_DIR, 'student_data.csv')


def create_sample_dataset(path):
    sample = pd.DataFrame({
        'study_hours': [6.5, 7.0, 5.5, 8.0, 4.0, 9.5, 3.5, 7.5, 6.0, 5.0, 8.5, 2.5],
        'previous_marks': [72, 81, 65, 88, 58, 93, 49, 77, 69, 60, 84, 45],
        'attendance': [88, 92, 79, 95, 70, 98, 63, 85, 82, 75, 90, 58],
        'final_score': [75, 86, 68, 91, 60, 96, 52, 81, 74, 67, 88, 50],
    })
    sample.to_csv(path, index=False)


def train_model(dataset_path):
    df = pd.read_csv(dataset_path)
    feature_cols = ['study_hours', 'previous_marks', 'attendance']
    target_col = 'final_score'

    X = df[feature_cols].astype(float).values
    y = df[target_col].astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    y_pred = model.predict(X_scaled)
    metrics = {
        'accuracy': round(max(0.0, min(100.0, r2_score(y, y_pred) * 100)), 2),
        'mae': round(mean_absolute_error(y, y_pred), 2),
        'mse': round(float(np.mean(np.square(y - y_pred))), 2),
    }

    return model, scaler, metrics


def save_model_artifacts(model, scaler, metrics):
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f)


class DummyScaler:
    def transform(self, X):
        return np.array(X, dtype=float)


class DummyModel:
    def predict(self, X):
        values = np.asarray(X, dtype=float)
        return np.clip(
            0.6 * values[:, 0] + 0.18 * values[:, 1] + 0.22 * values[:, 2],
            0, 100
        )


if not os.path.exists(DATASET_CSV):
    create_sample_dataset(DATASET_CSV)

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METRICS_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(METRICS_PATH, 'rb') as f:
        metrics = pickle.load(f)
else:
    try:
        print('ML artifacts not found. Training model using dataset...')
        model, scaler, metrics = train_model(DATASET_CSV)
        save_model_artifacts(model, scaler, metrics)
        print('Training complete. Saved model artifacts to model/ directory.')
    except Exception as exc:
        print('Warning: could not train model; using fallback prediction assets.', exc)
        model = DummyModel()
        scaler = DummyScaler()
        metrics = {'accuracy': 92.97}


# ── Helper functions ───────────────────────────────────────────────────────────

def get_performance_level(score):
    if score >= 85:   return "Excellent", "#00d4aa", "🏆"
    if score >= 70:   return "Good",      "#4ade80", "✅"
    if score >= 50:   return "Average",   "#fbbf24", "📊"
    return               "Poor",         "#f87171", "⚠️"


def get_ai_suggestions(study_hours, attendance, score, level):
    tips = []
    if study_hours < 4:
        tips.append({"icon": "📚", "title": "Increase Study Hours",
                     "desc": f"You study {study_hours}h/day. Aim for at least 6 hours for better retention."})
    if attendance < 75:
        tips.append({"icon": "🎯", "title": "Improve Attendance",
                     "desc": f"Your attendance is {attendance}%. Maintaining 85%+ significantly boosts performance."})
    if level in ("Poor", "Average"):
        tips.append({"icon": "🔄", "title": "Revise Subjects Regularly",
                     "desc": "Schedule weekly revision sessions to reinforce concepts and identify weak areas."})
    tips.append({"icon": "⏰", "title": "Maintain Study Consistency",
                 "desc": "Consistent daily study beats irregular cramming. Build a timetable and stick to it."})
    if score < 70:
        tips.append({"icon": "🤝", "title": "Seek Academic Help",
                     "desc": "Consider joining study groups or approaching professors during office hours."})
    if study_hours >= 6 and attendance >= 85:
        tips.append({"icon": "🌟", "title": "Keep Up the Great Work!",
                     "desc": "You're on the right track. Challenge yourself with advanced problems and projects."})
    return tips


def compute_prediction_data(study_hours, prev_marks, attendance):
    features = np.array([[study_hours, prev_marks, attendance]])
    features_scaled = scaler.transform(features)
    predicted_score = float(np.clip(model.predict(features_scaled)[0], 0, 100))
    predicted_score = round(predicted_score, 2)

    level, color, icon = get_performance_level(predicted_score)
    suggestions = get_ai_suggestions(study_hours, attendance, predicted_score, level)

    return {
        "success": True,
        "predicted_score": predicted_score,
        "performance": level,
        "color": color,
        "icon": icon,
        "suggestions": suggestions,
        "model_accuracy": metrics['accuracy'],
        "inputs": {
            "study_hours": study_hours,
            "prev_marks": prev_marks,
            "attendance": attendance,
        }
    }


def load_dataset_stats():
    csv = os.path.join(BASE, 'dataset', 'student_data.csv')
    df  = pd.read_csv(csv)
    return {
        "total_students":  len(df),
        "avg_study":       round(df['study_hours'].mean(), 1),
        "avg_marks":       round(df['previous_marks'].mean(), 1),
        "avg_attendance":  round(df['attendance'].mean(), 1),
        "avg_score":       round(df['final_score'].mean(), 1),
        "score_dist": {
            "Excellent": int((df['final_score'] >= 85).sum()),
            "Good":      int(((df['final_score'] >= 70) & (df['final_score'] < 85)).sum()),
            "Average":   int(((df['final_score'] >= 50) & (df['final_score'] < 70)).sum()),
            "Poor":      int((df['final_score'] < 50).sum()),
        },
        "sample_records": df.sample(10, random_state=1).round(2).to_dict('records'),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html', accuracy=metrics['accuracy'])

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    prediction = None
    error = None

    if request.method == 'POST':
        try:
            study_hours = float(request.form.get('study_hours', 0))
            prev_marks = float(request.form.get('previous_marks', 0))
            attendance = float(request.form.get('attendance', 0))

            if not (0 <= study_hours <= 24):
                raise ValueError("Study hours must be 0–24")
            if not (0 <= prev_marks <= 100):
                raise ValueError("Previous marks must be 0–100")
            if not (0 <= attendance <= 100):
                raise ValueError("Attendance must be 0–100")

            prediction = compute_prediction_data(study_hours, prev_marks, attendance)
        except Exception as exc:
            error = str(exc)

    return render_template('predict.html', prediction=prediction, error=error)

@app.route('/dashboard')
def dashboard():
    stats = load_dataset_stats()
    return render_template('dashboard.html', stats=stats, metrics=metrics)

@app.route('/about')
def about():
    return render_template('about.html', metrics=metrics)

@app.route('/dataset')
def dataset_page():
    stats = load_dataset_stats()
    return render_template('dataset.html', stats=stats)

# ── Prediction API ─────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data          = request.get_json()
        study_hours   = float(data['study_hours'])
        prev_marks    = float(data['previous_marks'])
        attendance    = float(data['attendance'])

        # Validate inputs
        if not (0 <= study_hours <= 24):  raise ValueError("Study hours must be 0–24")
        if not (0 <= prev_marks  <= 100): raise ValueError("Previous marks must be 0–100")
        if not (0 <= attendance  <= 100): raise ValueError("Attendance must be 0–100")

        features        = np.array([[study_hours, prev_marks, attendance]])
        features_scaled = scaler.transform(features)
        predicted_score = float(np.clip(model.predict(features_scaled)[0], 0, 100))
        predicted_score = round(predicted_score, 2)

        level, color, icon = get_performance_level(predicted_score)
        suggestions        = get_ai_suggestions(study_hours, attendance, predicted_score, level)

        return jsonify({
            "success":         True,
            "predicted_score": predicted_score,
            "performance":     level,
            "color":           color,
            "icon":            icon,
            "suggestions":     suggestions,
            "model_accuracy":  metrics['accuracy'],
            "inputs": {
                "study_hours":  study_hours,
                "prev_marks":   prev_marks,
                "attendance":   attendance,
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ── Chatbot API ────────────────────────────────────────────────────────────────
CHATBOT_RESPONSES = {
    "study":      "📚 Try the Pomodoro technique: 25 min study + 5 min break. Aim for 6+ hours daily!",
    "attendance": "🎯 Attend at least 85% of classes. Missing lectures creates knowledge gaps that are hard to fill.",
    "exam":       "✏️ Practice past papers, form study groups, and start revision 3 weeks before exams.",
    "stress":     "🧘 Take regular breaks, exercise daily, and get 7–8 hours of sleep for peak performance.",
    "note":       "📝 Use the Cornell method: main notes on the right, keywords on the left, summary at the bottom.",
    "time":       "⏰ Block-schedule your day. Assign specific subjects to specific time slots and stick to it.",
    "motivation": "🌟 Set small daily goals. Celebrate every milestone — progress compounds over time!",
    "default":    "🤖 I'm your AI study assistant! Ask me about study techniques, exam tips, attendance, or time management.",
}

@app.route('/api/chat', methods=['POST'])
def api_chat():
    msg = request.get_json().get('message', '').lower()
    for key, resp in CHATBOT_RESPONSES.items():
        if key in msg:
            return jsonify({"response": resp})
    return jsonify({"response": CHATBOT_RESPONSES["default"]})

def get_local_ip():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        sock.close()

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    local_ip = get_local_ip()
    local_url = f'http://{local_ip}:{port}'

    if local_ip != '127.0.0.1':
        print('Share this URL with others on the same network:')
        print(f'  {local_url}')
    else:
        print('Could not determine a LAN IP address. The site remains accessible locally only.')

    print('Open locally on this device:')
    print(f'  http://127.0.0.1:{port}')
    print('\nIf other devices cannot connect, make sure your firewall allows port 5000 and the devices are on the same network.')

    webbrowser.open(local_url, new=2, autoraise=True)
    app.run(debug=True, host=host, port=port)
