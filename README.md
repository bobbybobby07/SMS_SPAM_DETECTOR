# SMS Spam Detector

A small Flask web app that classifies an SMS message as **Spam** or **Not Spam**
using a **Multinomial Naive Bayes** model (scikit-learn) on bag-of-words features.

## Project structure

```
├── app.py            # Flask app (web page + JSON API)
├── train_model.py    # Trains the model and writes model.pkl
├── model.pkl         # Trained pipeline (CountVectorizer + MultinomialNB)
├── requirements.txt
├── Dockerfile
├── templates/index.html
├── static/style.css
└── tests/test_app.py
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py            # optional: model.pkl is already included
python app.py
```

Open http://localhost:5000.

## How it works

1. `train_model.py` downloads the SMS Spam Collection dataset (5,574 messages), removes duplicates,
   and splits it 80/20.
2. A pipeline of `CountVectorizer` (unigrams and bigrams, English stop words removed) and
   `MultinomialNB(alpha=0.1)` is trained. On the held-out test set it reaches about 98.7% accuracy.
3. The final model is refit on all data and saved to `model.pkl`.
4. `app.py` loads the pipeline once at startup and classifies each submitted message.

## API

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message": "WINNER!! Claim your free prize now"}'
# {"confidence": 99.9, "prediction": "Spam"}
```

Other routes: `GET /health` returns `{"status": "ok"}`.

## Tests

```bash
pytest
```

## Docker

```bash
docker build -t sms-spam-detector .
docker run -p 5000:5000 sms-spam-detector
```

The container serves the app with gunicorn on port 5000 (override with `-e PORT=8080`).
