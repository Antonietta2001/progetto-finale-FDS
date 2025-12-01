# ml_server.py - INFERENZA V9 (MASSIMA STABILITÀ)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from threading import Timer
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.models import load_model
import sys
import os
import time
import random
from typing import List

# --- CONFIGURAZIONE GLOBALE ---
MODEL_FILE = 'lstm_market_heartbeat_model_v8_risk_tuned.keras'
THRESHOLD_FILE = 'optimal_threshold_v8_risk_tuned.npy'
MEAN_FILE = 'normalization_mean_v7_essential.npy'
STD_FILE = 'normalization_std_v7_essential.npy'

SEQUENCE_LENGTH = 10 
FEATURE_COUNT = 18 
FINAL_POS_WEIGHT = 153.75 


LSTM_MODEL = None
NORM_MEAN = None
NORM_STD = None
RISK_THRESHOLD = None
SENTIMENT_CACHE = {"fear_greed_index": 50.0, "twitter_sentiment": 0.0, "last_update": time.time()}
SENTIMENT_UPDATE_INTERVALS_SECONDS = 600

# --- FUNZIONI CUSTOM (Necessarie per la deserializzazione) ---

class AttentionLayer(layers.Layer):
    """Necessaria per caricare l'Attention Layer custom"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def call(self, inputs):
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

def weighted_binary_crossentropy(pos_weight):
    """Necessaria per caricare la Loss custom"""
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        weight = y_true * pos_weight + (1 - y_true) * 1.0
        weighted_bce = weight * bce
        return tf.reduce_mean(weighted_bce)
    return loss

# --- SCHEMA DATI ALLINEATO ---
class FeatureVector(BaseModel):
    """
    Rappresenta il vettore di 18 feature numeriche.
    Il frontend DEVE inviare una lista di 18 float sotto la chiave 'data'.
    """
    data: List[float] = Field(min_length=FEATURE_COUNT, max_length=FEATURE_COUNT)
    
class PredictRequest(BaseModel):
    """Richiesta con la sequenza di 10 timesteps."""
    sequence: List[FeatureVector] = Field(min_length=SEQUENCE_LENGTH, max_length=SEQUENCE_LENGTH)

# --- LOGICA DI CARICAMENTO E CACHING ---

def load_ml_assets():
    global LSTM_MODEL, NORM_MEAN, NORM_STD, RISK_THRESHOLD
    
    custom_objects = {'loss': weighted_binary_crossentropy(FINAL_POS_WEIGHT), 'AttentionLayer': AttentionLayer}

    if os.path.exists(MODEL_FILE):
        try:
            print(f"🔄 Tentativo di caricamento modello V8...")
            LSTM_MODEL = load_model(MODEL_FILE, custom_objects=custom_objects)
            NORM_MEAN = np.load(MEAN_FILE)
            NORM_STD = np.load(STD_FILE)
            RISK_THRESHOLD = np.load(THRESHOLD_FILE)[()]
            
            print(f"✅ Modello V8 caricato. Soglia di Rischio: {RISK_THRESHOLD:.4f}")
        except Exception as e:
            print(f"❌ ERRORE CRITICO nel caricamento degli asset: {e}")
            sys.exit(1)
    else:
        print(f"❌ ERRORE: File modello non trovato ({MODEL_FILE}).")
        sys.exit(1)

def fetch_fear_greed_api(): return random.randint(15, 85)
def fetch_twitter_sentiment(): 
    calculation = ((SENTIMENT_CACHE["fear_greed_index"] - 50) / 50 * 0.5 + random.uniform(-0.3, 0.3))
    return round(calculation, 4)

def update_sentiment_cache():
    global SENTIMENT_CACHE
    fng_index = fetch_fear_greed_api()
    twitter_sentiment = fetch_twitter_sentiment()
    
    SENTIMENT_CACHE.update({
        "fear_greed_index": float(fng_index),
        "twitter_sentiment": float(twitter_sentiment),
        "last_update": time.time()
    })
    
    timer = Timer(SENTIMENT_UPDATE_INTERVALS_SECONDS, update_sentiment_cache)
    timer.daemon = True
    timer.start()

# --- AVVIO SERVER E LOGICA PRINCIPALE ---

app = FastAPI(title="Market Heartbeat ML Predictor V8")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """Esegue il caricamento del modello una volta sola all'avvio."""
    load_ml_assets()
    update_sentiment_cache()

@app.post("/predict_risk")
async def predict_risk(data: PredictRequest):
    
    # Il controllo della validazione della forma e del contenuto numerico è gestito da Pydantic
    
    # 1. Ricostruzione dell'Array di Input
    combined_input = np.array([s.data for s in data.sequence], dtype=np.float32)
    
    # 2. Protezione da NaN/Inf (Safety Check)
    combined_input = np.nan_to_num(
        combined_input, 
        nan=0.0, 
        posinf=1e10,
        neginf=-1e10 
    )
    
    # 3. Aggiornamento Sentiment Live (Tutti i 10 Timesteps)
    live_fng = SENTIMENT_CACHE["fear_greed_index"]
    live_twitter = SENTIMENT_CACHE["twitter_sentiment"]
    
    # Sovrascrive le feature 16 e 17 (indice 15 e 16) per tutti i 10 timesteps
    combined_input[:, FEATURE_COUNT-2] = live_fng 
    combined_input[:, FEATURE_COUNT-1] = live_twitter 
    
    # 4. Normalizzazione e Predizione
    normalized_input = (combined_input - NORM_MEAN) / (NORM_STD + 1e-6)
    lstm_input = normalized_input.reshape(1, SEQUENCE_LENGTH, FEATURE_COUNT)
    
    prediction_prob = LSTM_MODEL.predict(lstm_input, verbose=0)[0][0]
    
    # 5. Determinazione del Livello di Rischio (Usa RISK_THRESHOLD caricato)
    RISK_THRESHOLD_LIVE = RISK_THRESHOLD 
    
    if prediction_prob >= RISK_THRESHOLD_LIVE: 
        risk_level = "CRITICAL_SELL" 
    elif prediction_prob >= (RISK_THRESHOLD_LIVE * 0.95):
        risk_level = "HIGH_ALERT" 
    elif prediction_prob >= (RISK_THRESHOLD_LIVE * 0.85):
        risk_level = "MEDIUM"
    else: 
        risk_level = "NORMAL"

    return {
        "probability_of_crash": float(prediction_prob), 
        "risk_level_predicted": risk_level,
        "risk_threshold_tuned": float(RISK_THRESHOLD_LIVE),
        "fng_index_live": SENTIMENT_CACHE["fear_greed_index"],
        "twitter_sentiment_live": SENTIMENT_CACHE["twitter_sentiment"]
    }
