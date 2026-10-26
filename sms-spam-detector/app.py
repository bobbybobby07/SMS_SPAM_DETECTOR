"""Flask web app for SMS spam detection (Multinomial Naive Bayes)."""
import os
import pickle

from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
MAX_LENGTH = 1000

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def classify(text):
    """Return ('Spam' | 'Not Spam', confidence 0-100) for a message."""
    proba = model.predict_proba([text])[0]
    is_spam = proba[1] >= 0.5
    confidence = proba[1] if is_spam else proba[0]
    return ("Spam" if is_spam else "Not Spam"), round(float(confidence) * 100, 1)


@app.route("/", methods=["GET", "POST"])
def index():
    message, result, confidence, error = "", None, None, None
    if request.method == "POST":
        message = request.form.get("message", "").strip()
        if not message:
            error = "Enter a message to check."
        elif len(message) > MAX_LENGTH:
            error = f"Messages can be up to {MAX_LENGTH} characters."
        else:
            result, confidence = classify(message)
    return render_template(
        "index.html", message=message, result=result, confidence=confidence, error=error
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify(error="'message' is required"), 400
    if len(message) > MAX_LENGTH:
        return jsonify(error=f"'message' must be at most {MAX_LENGTH} characters"), 400
    label, confidence = classify(message)
    return jsonify(prediction=label, confidence=confidence)


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
