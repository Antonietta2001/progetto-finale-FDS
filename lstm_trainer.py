# lstm_trainer.py - Versione Finale AUC Optimization (10 Feature)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.utils import class_weight
from tensorflow.keras import regularizers

# --- CONFIGURAZIONE ---
FEATURE_COUNT = 10 # CORRETTO: 10 Feature totali
LSTM_UNITS = 128    
EPOCHS = 50         
BATCH_SIZE = 32
EARLY_STOP_PATIENCE = 7 
OPTIMIZER = Adam(learning_rate=0.0001, clipnorm=1.0) 
THRESHOLD = 0.75 
DROPOUT_RATE = 0.5 

# Carica i dati (omesso codice per brevità)
try:
    X = np.load('X_train.npy'); Y = np.load('Y_train.npy')
except FileNotFoundError: raise FileNotFoundError("ERRORE: Eseguire data_processor.py prima."); exit()

# 1. Suddivisione Training, Validation, Test Set
train_size = int(0.7 * len(X)); val_size = int(0.15 * len(X))
X_train, X_temp = X[:train_size], X[train_size:]
Y_train, Y_temp = Y[:train_size], Y[train_size:]
X_val, X_test = X_temp[:val_size], X_temp[val_size:]
Y_val, Y_test = Y_temp[:val_size], Y_temp[val_size:] 

# 2. CALCOLO DEI PESI DI CLASSE
y_ints = Y_train.astype(int)
class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)
class_weights = dict(enumerate(class_weights_array))

# 3. Definizione del Modello LSTM
_, TIME_STEPS, FEATURES = X_train.shape 
def build_lstm_model(timesteps, features):
    model = Sequential([
        LSTM(
            LSTM_UNITS, 
            activation='tanh', 
            input_shape=(timesteps, features),
            kernel_regularizer=regularizers.l2(0.001) 
        ),
        Dropout(DROPOUT_RATE), 
        Dense(1, activation='sigmoid') 
    ])
    model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_lstm_model(TIME_STEPS, FEATURES)
# ... (Omesse fasi di training e valutazione) ...

# 4. Addestramento del Modello
callbacks = [EarlyStopping(monitor='val_loss', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]

history = model.fit(
    X_train, Y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_data=(X_val, Y_val), class_weight=class_weights, callbacks=callbacks, verbose=1
)

# 5. Valutazione sul Test Set e Salvataggio
Y_pred_proba = model.predict(X_test).flatten()
Y_pred = (Y_pred_proba > THRESHOLD).astype(int)
auc_roc = roc_auc_score(Y_test, Y_pred_proba)

print("\n--- Validazione Finale ---")
print(f"1. AUC-ROC Score: {auc_roc:.4f}")
print("2. Report di Classificazione:\n", classification_report(Y_test, Y_pred, target_names=['Non-Crash (0)', 'Crash (1)']))

MODEL_FILE = 'lstm_market_heartbeat_model.h5'
model.save(MODEL_FILE)