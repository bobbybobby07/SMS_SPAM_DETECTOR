# SMS Spam Detector

A Flask web app that estimates whether a text message resembles spam. Paste a message into the browser-based checker to get a **Spam** or **Not Spam** prediction and the model's confidence. The app also provides a JSON API for programmatic predictions.

## Features

- Interactive checker with sample messages and a 1,000-character input limit.
- An `/about` page explaining the model and its limitations.
- JSON prediction endpoint and a health-check endpoint.
- Pre-trained scikit-learn model included, so the app can run without retraining.
- Docker image served with Gunicorn.
- Messages are processed for the request and are not stored by the app.

## How it works

The training script downloads the SMS Spam Collection dataset (5,574 messages), removes duplicate entries, and creates an 80/20 train/test split. A scikit-learn pipeline turns text into unigram and bigram counts using `CountVectorizer`, then classifies the counts with `MultinomialNB(alpha=0.1)`. The documented held-out accuracy is about 98.7%; this is a dataset-specific result, not a guarantee for new messages.

The final pipeline is refit on the full dataset and saved as `sms-spam-detector/model.pkl`. At startup, Flask loads that model once. For each valid message, the app returns the predicted label and the probability assigned to that label, rounded to one decimal place.

## Project layout

```text
sms-spam-detector/
  app.py                  Flask pages and JSON endpoints
  model.pkl               Trained text-classification pipeline
  train_model.py          Download data and train the model
  requirements.txt        Python dependencies
  Dockerfile              Container configuration
  templates/
    index.html             Message checker
    about.html             Project and model information
  static/style.css         Page styles
  tests/test_app.py        Route and prediction tests
```

## Run locally

Python 3 and pip are required. From the repository root:

```powershell
cd sms-spam-detector
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://localhost:5000>. On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. To retrain the model, run `python train_model.py`; this downloads the dataset, so an internet connection is required. Retraining is optional because `model.pkl` is included.

## HTTP endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Message checker |
| `GET` | `/about` | Project and model information |
| `POST` | `/api/predict` | Classify a JSON message |
| `GET` | `/health` | Return `{"status":"ok"}` |

Example API request:

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER! Claim your free prize now"}'
```

Example response:

```json
{"confidence": 99.9, "prediction": "Spam"}
```

The API returns HTTP 400 for a missing, blank, or over-limit message. Predictions are best-effort: legitimate texts can be flagged, and spam can be missed. Do not rely on this model alone to decide whether a link or request is trustworthy.

## Tests

From `sms-spam-detector/`, install the dependencies and run:

```bash
python -m pytest
```

## Docker

From `sms-spam-detector/`:

```bash
docker build -t sms-spam-detector .
docker run --rm -p 5000:5000 sms-spam-detector
```

The container listens on port 5000. Set `PORT` to use another internal port.
