import logging
import os
import glob
import pickle
import pandas as pd
import time
import json
import re
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Body, HTTPException
from pydantic import BaseModel
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fastapi.middleware.cors import CORSMiddleware

# Logging setup
logger = logging.getLogger("inference_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# WebSocket log broadcast setup
active_websockets = []

def broadcast_log(msg):
    import asyncio
    for ws in list(active_websockets):
        try:
            asyncio.create_task(ws.send_text(msg))
        except Exception:
            try:
                active_websockets.remove(ws)
            except Exception:
                pass

def log_and_broadcast(msg):
    logger.info(msg)
    broadcast_log(msg)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ML / Transformers
st_model = SentenceTransformer("xlm-roberta-large")
log_and_broadcast("AI. ROBERTA large model učitan.")
TRAIN_DATA_DIR = "./train_data"
MODEL_PATH = "./trained_model.pkl"
VECTORIZER_PATH = "./vectorizer.pkl"
LABEL_ENCODER_PATH = "./label_encoder.pkl"
METRICS_FILE = "/tmp/training_batches/last_training_metrics.json"
os.makedirs(TRAIN_DATA_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

current_vectorizer = None
current_clf = None
label_encoder = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        current_vectorizer, current_clf = pickle.load(f)
    log_and_broadcast("TFIDF+LogReg model učitan iz pickle-a.")

if os.path.exists(VECTORIZER_PATH):
    with open(VECTORIZER_PATH, "rb") as f:
        current_vectorizer = pickle.load(f)
    log_and_broadcast("Vectorizer učitan.")

if os.path.exists(LABEL_ENCODER_PATH):
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)
    log_and_broadcast("Label encoder učitan.")

# Regexes (hardcoded or loaded from local regex_common.py)
from regex_common import (
    DOC_NUMBER_PATTERNS,
    OIB_PATTERN,
    COUNTRY_VAT_REGEX,
    VAT_CANDIDATE_PATTERN,
    DATE_PATTERNS,
    extract_doc_number,
    extract_oib,
    extract_all_oibs,
    extract_vat_number,
    extract_all_vats,
    extract_dates,
    extract_invoice_date,
    extract_due_date,
)

def find_amount_near_keywords(text, keywords):
    lines = text.split("\n")
    amount_pattern = r"(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})"
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                matches = re.findall(amount_pattern, line.replace(" ", ""))
                if matches:
                    return matches[-1].replace(",", ".")
                if i + 1 < len(lines):
                    matches = re.findall(amount_pattern, lines[i+1].replace(" ", ""))
                    if matches:
                        return matches[-1].replace(",", ".")
                if i > 0:
                    matches = re.findall(amount_pattern, lines[i-1].replace(" ", ""))
                    if matches:
                        return matches[-1].replace(",", ".")
    return None

# Definiraj Pydantic model za input
class ClassifyRequest(BaseModel):
    text: str
    label_examples: Dict[str, List[str]] = {}

@app.post("/classify")
async def classify_text(data: ClassifyRequest):
    log_and_broadcast(f"Primljen zahtjev za klasifikaciju: {data.text[:50]}...")
    threshold = 0.45

    if current_vectorizer and current_clf:
        X = current_vectorizer.transform([data.text])
        proba = current_clf.predict_proba(X)[0]
        classes = current_clf.classes_
        label_scores = dict(zip(classes, proba))
        label_scores = {str(k): float(v) for k, v in label_scores.items()}
        best_label = str(classes[proba.argmax()])
        best_score = float(proba.max())
        if best_score < threshold:
            best_label = "NEPOZNATO"
        result = {
            "label_scores": label_scores,
            "best_label": best_label,
            "best_score": best_score,
            "source": "ML model"
        }
        log_and_broadcast(f"Rezultat (ML model): {result}")
        return result

    doc_embedding = st_model.encode(data.text, convert_to_tensor=True)
    label_scores = {}
    for label, examples in data.label_examples.items():
        if not examples:
            label_scores[label] = 0.0
            continue
        example_embeddings = st_model.encode(examples, convert_to_tensor=True)
        similarity = util.cos_sim(doc_embedding, example_embeddings).mean().item()
        label_scores[label] = float(similarity)
    sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_labels[0]
    best_label = str(best_label)
    best_score = float(best_score)
    if best_score < threshold:
        best_label = "NEPOZNATO"
    result = {
        "label_scores": label_scores,
        "best_label": best_label,
        "best_score": best_score,
        "source": "transformers"
    }
    log_and_broadcast(f"Rezultat (sentence-transformers): {result}")
    return result

@app.post("/api/new_training_data")
async def receive_training_data(file: UploadFile = File(...)):
    save_path = os.path.join(TRAIN_DATA_DIR, file.filename)
    with open(save_path, "wb") as f:
        f.write(await file.read())
    log_and_broadcast(f"Primljen novi trening dataset: {save_path}")
    return {"message": f"Trening podaci spremljeni kao {save_path}"}

@app.post("/api/train_model")
async def train_model():
    log_and_broadcast("Pokrećem treniranje modela na novim podacima...")
    train_files = glob.glob(os.path.join(TRAIN_DATA_DIR, "*.csv"))
    if not train_files:
        return {"message": "Nema trening podataka!"}

    texts, labels = [], []
    for file in train_files:
        try:
            df = pd.read_csv(file)
            texts.extend(df["text"].tolist())
            labels.extend(df["label"].tolist())
        except Exception as e:
            log_and_broadcast(f"Greška pri čitanju {file}: {e}")

    if not texts:
        return {"message": "Nema valjanih uzoraka za treniranje!"}

    start_time = time.time()

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(texts)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    clf = LogisticRegression(max_iter=200)
    clf.fit(X, y)

    training_time = time.time() - start_time

    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average=None)
    classes = le.classes_

    precision_dict = {cls: float(p) for cls, p in zip(classes, precision)}
    recall_dict = {cls: float(r) for cls, r in zip(classes, recall)}
    f1_dict = {cls: float(f) for cls, f in zip(classes, f1)}

    with open(MODEL_PATH, "wb") as f:
        pickle.dump((vectorizer, clf), f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    global current_vectorizer, current_clf, label_encoder
    current_vectorizer = vectorizer
    current_clf = clf
    label_encoder = le

    log_and_broadcast(f"Model istreniran i spremljen na {len(texts)} uzoraka. Trajanje treninga: {training_time:.2f}s")

    metrics = {
        "accuracy": accuracy,
        "precision": precision_dict,
        "recall": recall_dict,
        "f1_score": f1_dict,
        "loss": None,
        "training_time_seconds": training_time,
        "epochs": 1
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    return {
        "message": f"Model istreniran i spremljen na {len(texts)} uzoraka.",
        "metrics": metrics
    }

@app.get("/api/ml/metrics")
async def get_ml_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return metrics
    return {"error": "Metrike nisu dostupne"}, 404

@app.post("/parse-document/")
async def parse_document(data: dict = Body(...)):
    text = data.get("text", "")
    if not text:
        return {"error": "No text provided"}

    word_count = len(text.split())
    char_count = len(text)

    parsed_fields = {
        "invoice_number": extract_doc_number(text),
        "oib": extract_oib(text),
        "vat_number": extract_vat_number(text),
        **extract_dates(text),
    }

    amount_keywords = ["Ukupno", "Za platiti", "Total"]
    amount = find_amount_near_keywords(text, amount_keywords)
    parsed_fields["amount_total"] = amount

    vat = parsed_fields.get("vat_number")
    valid_eu_vat = False
    if vat:
        for pattern in COUNTRY_VAT_REGEX.values():
            if re.fullmatch(pattern, vat):
                valid_eu_vat = True
                break

    supplier_name = None
    if not valid_eu_vat:
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "dobavljač" in line.lower() and i + 1 < len(lines):
                supplier_name = lines[i + 1].strip()
                break
        if not supplier_name:
            supplier_name = "Strani dobavljač"
    parsed_fields["supplier_name"] = supplier_name

    result = {
        "char_count": char_count,
        "word_count": word_count,
        "parsed_fields": parsed_fields,
    }
    log_and_broadcast(f"Parsiran dokument: {result}")
    return result

@app.websocket("/ws/training-logs")
async def training_logs_ws(websocket: WebSocket):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        try:
            active_websockets.remove(websocket)
        except Exception:
            pass
