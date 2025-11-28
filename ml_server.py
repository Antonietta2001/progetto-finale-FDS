# ml_server.py - Inferenza in Tempo Reale (Sincronizzato 10 Feature)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from threading import Timer
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential 
from typing import List
import time
import random
import os

# --- CONFIGURAZIONE ---
app = FastAPI(title="Market Heartbeat ML Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

MODEL_FILE = 'lstm_market_heartbeat_model.h5'
MEAN_FILE = 'normalization_mean.npy'; STD_FILE = 'normalization_std.npy'

LSTM_MODEL = None
NORM_MEAN = None
NORM_STD = None
SEQUENCE_LENGTH = 5 # Corrisponde al trainer
FEATURE_COUNT = 10 # Corrisponde al trainer

# [Omessa logica di caricamento asset e polling sentiment]

def create_dummy_model():
    model = Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(SEQUENCE_LENGTH, FEATURE_COUNT)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]); model.compile(optimizer='adam', loss='binary_crossentropy'); return model

def load_ml_assets():
    global LSTM_MODEL, NORM_MEAN, NORM_STD
    if os.path.exists(MODEL_FILE) and os.path.exists(MEAN_FILE):
        try:
            LSTM_MODEL = load_model(MODEL_FILE); NORM_MEAN = np.load(MEAN_FILE); NORM_STD = np.load(STD_FILE)
            print("✅ Modello addestrato e parametri di normalizzazione caricati.")
        except Exception as e:
            LSTM_MODEL = create_dummy_model(); NORM_MEAN = np.zeros(FEATURE_COUNT); NORM_STD = np.ones(FEATURE_COUNT)
    else:
        LSTM_MODEL = create_dummy_model(); NORM_MEAN = np.zeros(FEATURE_COUNT); NORM_STD = np.ones(FEATURE_COUNT)

# Sentiment Polling 
SENTIMENT_CACHE = {"fear_greed_index": 50, "twitter_sentiment": 0.0, "last_update": time.time()}
SENTIMENT_UPDATE_INTERVAL_SECONDS = 600
def fetch_fear_greed_api(): return random.randint(15, 85)
def fetch_twitter_sentiment(): 
    calculation = ((SENTIMENT_CACHE["fear_greed_index"] - 50) / 50 * 0.5 + random.uniform(-0.3, 0.3))
    return round(calculation, 2)
def update_sentiment_cache():
    global SENTIMENT_CACHE
    fng_index = fetch_fear_greed_api(); twitter_sentiment = fetch_twitter_sentiment()
    SENTIMENT_CACHE.update({"fear_greed_index": fng_index, "twitter_sentiment": twitter_sentiment, "last_update": time.time()})
    timer = Timer(SENTIMENT_UPDATE_INTERVALS_SECONDS, update_sentiment_cache); timer.daemon = True; timer.start()

load_ml_assets()
update_sentiment_cache()

# --- SCHEMA DATI AGGIORNATO (CORREZIONE FINALE) ---

class FeatureVector(BaseModel):
    # Dati inviati dal frontend (rimane lo schema originale del frontend)
    time: str; ofi: float; hawkes_intensity: float; spread_bps: float; imbalance_ratio: float; bid_depth: float; ask_depth: float; price: float 
    
class PredictRequest(BaseModel):
    sequence: List[FeatureVector]

@app.post("/predict_risk")
async def predict_risk(data: PredictRequest):
    if len(data.sequence) != SEQUENCE_LENGTH:
        raise HTTPException(status_code=400, detail=f"Richiesta sequenza errata.")

    # 1. Ricostruzione dell'Array di Input (10 FEATURE TOTALI)
    # Calcolo Live Proxy per Delta_OFI, Aggressive_Volume, etc. (Usando solo i dati ricevuti)
    
    # Calcolo Proxy live delle feature aggiuntive
    ofi_sequence = np.array([s.ofi for s in data.sequence])
    delta_ofi = ofi_sequence[-1] - ofi_sequence[-2] if SEQUENCE_LENGTH >= 2 else 0.0
    
    # Range Volatility Proxy: usiamo lo spread bps del momento come proxy
    avg_spread_bps = np.mean([s.spread_bps for s in data.sequence])
    
    # Volume Totale Aggressivo Proxy (usiamo OFI assoluto + Depth media)
    avg_depth = np.mean([(s.bid_depth + s.ask_depth) / 2 for s in data.sequence])
    aggressive_volume_proxy = np.abs(ofi_sequence[-1]) + avg_depth
    
    # Imbalance Ratio Abs Proxy
    imbalance_ratio_abs = np.abs(ofi_sequence[-1]) / (np.abs(ofi_sequence[-1]) + avg_depth + 1e-6)

    combined_input = np.array([[
        s.ofi, 
        s.hawkes_intensity, 
        s.price,                       
        SENTIMENT_CACHE["fear_greed_index"], 
        SENTIMENT_CACHE["twitter_sentiment"],
        s.spread_bps,                  
        delta_ofi, 
        avg_spread_bps, 
        aggressive_volume_proxy,
        imbalance_ratio_abs
    ] for s in data.sequence])
    
    # 2. Normalizzazione e Predizione
    normalized_input = (combined_input - NORM_MEAN) / (NORM_STD + 1e-6)
    lstm_input = normalized_input.reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
    prediction_prob = LSTM_MODEL.predict(lstm_input, verbose=0)[0][0]
    
    # 3. Determinazione del Livello di Rischio (Soglia)
    THRESHOLD_INFERENZA = 0.75
    
    if prediction_prob >= 0.85: risk_level = "Critical"
    elif prediction_prob >= THRESHOLD_INFERENZA: risk_level = "High"
    elif prediction_prob >= 0.40: risk_level = "Medium"
    else: risk_level = "Normal"

    return {
        "probability_of_crash": float(prediction_prob), 
        "risk_level_predicted": risk_level,
        "fear_greed_index": SENTIMENT_CACHE["fear_greed_index"],
        "twitter_sentiment": SENTIMENT_CACHE["twitter_sentiment"]
    }