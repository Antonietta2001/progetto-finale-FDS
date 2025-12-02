import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix
import sys
from tensorflow.keras.optimizers import Adam 

# --- CONFIGURAZIONE FILE E RISULTATI FINALI ---

# Parametri modello e dati (Assicurati che i nomi dei file siano corretti)
MODEL_FILE = 'lstm_market_heartbeat_model_v8_risk_tuned.keras'
THRESHOLD_FILE = 'optimal_threshold_v8_risk_tuned.npy'
MEAN_FILE = 'normalization_mean_v7_essential.npy' 
STD_FILE = 'normalization_std_v7_essential.npy'
X_TRAIN_SOURCE_FILE = 'X_train_v6_crash.npy' 
Y_TRAIN_SOURCE_FILE = 'Y_train_v6_crash.npy'

# Valori definitivi del training V8
FINAL_POS_WEIGHT = 153.75 

# --- LOGICA ECONOMICA (Parametri di Utilità) ---
COST_PER_FP = 0.50     # Costo per Falso Allarme (commissioni/slippage per un'azione non necessaria)
LOSS_PER_FN = 100.00   # Perdita stimata per Crash Mancato
PROFIT_PER_TP = 50.00  # Profitto/Perdita Evitata per Crash Previsto

# --- FUNZIONI CUSTOM (Necessarie per caricare il modello) ---

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

# --- CLASSE DI BACKTESTING ---

class HFTBacktester:
    def __init__(self):
        self.model = None
        self.mean = None
        self.std = None
        self.risk_threshold = 0.5
        self.X_test = None
        self.Y_test = None
        self.load_components()

    def load_components(self):
        print("▶️ Caricamento Componenti per Simulazione...")
        try:
            # Carica Modello
            self.model = keras.models.load_model(
                MODEL_FILE,
                custom_objects={'loss': weighted_binary_crossentropy(FINAL_POS_WEIGHT), 'AttentionLayer': AttentionLayer}
            )
            # Carica Parametri
            self.risk_threshold = np.load(THRESHOLD_FILE)
            self.mean = np.load(MEAN_FILE)
            self.std = np.load(STD_FILE)
            
            # Carica Dati e Split per isolare il Test Set (stesso split 70/15/15 del trainer)
            X = np.load(X_TRAIN_SOURCE_FILE)
            Y = np.load(Y_TRAIN_SOURCE_FILE)
            
            train_size = int(0.70 * len(X))
            val_size = int(0.15 * len(X))
            self.X_test = X[train_size + val_size:]
            self.Y_test = Y[train_size + val_size:]
            
            print(f"✅ Setup completato. Test Set caricato: {len(self.X_test):,} campioni.")

        except FileNotFoundError as e:
            print(f"❌ ERRORE: File non trovato ({e.filename}). Assicurati che i nomi dei file siano esatti.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ERRORE durante il caricamento: {e}. Controlla la variabile FINAL_POS_WEIGHT.")
            sys.exit(1)

    def run_simulation(self):
        """Simula il trading sul Test Set utilizzando la predizione vettorializzata."""
        
        print(f"\n--- INIZIO SIMULAZIONE DI BACKTESTING VETTORIALIZZATA ---")
        
        # 1. Normalizzazione Vettoriale (Applicata all'intero set di test)
        print("1. Normalizzazione e predizione vettoriale...")
        X_normalized = (self.X_test - self.mean) / (self.std + 1e-8)
        
        # 2. Predizione in un unico batch (VELOCE!)
        Y_pred_proba = self.model.predict(X_normalized, verbose=1).flatten()
        
        # 3. Decisione finale (Applicazione della Soglia Risk-Tuned all'intero array)
        print(f"2. Applicazione soglia {self.risk_threshold:.4f}...")
        Y_predicted_labels = (Y_pred_proba >= self.risk_threshold).astype(int)
            
        # Analisi Finale: Calcolo Matrice di Confusione
        cm = confusion_matrix(self.Y_test, Y_predicted_labels)
        
        # L'ordine .ravel() è TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        
        # Riutilizzo dei costi definiti
        Total_FP_Cost = fp * COST_PER_FP
        Total_FN_Loss = fn * LOSS_PER_FN
        Total_TP_Profit = tp * PROFIT_PER_TP
        
        Net_Utility = Total_TP_Profit - Total_FN_Loss - Total_FP_Cost

        print("\n--- RISULTATI DELLA SIMULAZIONE ---")
        print(f"Soglia Risk-Tuned utilizzata: {self.risk_threshold:.4f}")
        
        # Confronta con i risultati del tuo training
        print(f"\n🔢 Matrice di Confusione (Simulazione - V8 Risultati Attesi):")
        print(f"   True Positives (TP): {tp:,} (Atteso: 3,006)")
        print(f"   False Negatives (FN): {fn:,} (Atteso: 4,507)")
        print(f"   False Positives (FP): {fp:,} (Atteso: 45,912)")
        
        print(f"\n💰 Analisi di Utilità Economica (Costi usati per simulazione):")
        print(f"   Costo totale Falsi Allarmi (FP): ${Total_FP_Cost:,.2f} ({fp:,} eventi)")
        print(f"   Costo totale Crash Mancati (FN): ${Total_FN_Loss:,.2f} ({fn:,} eventi)")
        print(f"   Profitto lordo Crash Previsti (TP): ${Total_TP_Profit:,.2f} ({tp:,} eventi)")
        print(f"\n   UTILITY ECONOMICA NETTA: ${Net_Utility:,.2f}")

        if Net_Utility > 0:
            print("\n✅ CONCLUSIONE: La strategia Risk-Tuned genera un'Utilità Economica Netta Positiva.")
        else:
            print("\n❌ CONCLUSIONE: L'Utilità Netta è Negativa, il costo dei Falsi Allarmi/FN supera i profitti/perdite evitate.")


if __name__ == '__main__':
    backtester = HFTBacktester()
    backtester.run_simulation()