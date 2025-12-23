# =========================
# LSTM-CNN TRAINER
# =========================

SEED = 60

import os
import random
import numpy as np
import tensorflow as tf

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

import json
import pandas as pd
import keras
from keras import layers, Model
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# CONFIG
H = 10
PREFIX = "CORRECT_H10"
TARGET_ACCURACY = 0.69

N_TRIALS = 8
EPOCHS = 40
SEARCH_EPOCHS = 20

HYPER = {
    "cnn_filters":  [32, 64],
    "lstm_units_1": [64, 96, 128],
    "lstm_units_2": [32, 64],
    "dropout":      [0.2, 0.3, 0.4],
    "initial_lr":   [1e-3, 5e-4, 2e-4],
    "batch_size":   [512, 1024],
}

SEQ_LEN = 3

PARQ_TRAIN = "data_train_CORRECT.parquet"
PARQ_OPT   = "data_optimize_CORRECT.parquet"
PARQ_TEST  = "data_test_CORRECT.parquet"

X_TRAIN_FILE = "X_train_CORRECT.npy"
Y_TRAIN_FILE = "Y_train_CORRECT.npy"
MEAN_FILE    = "normalization_mean_CORRECT.npy"
STD_FILE     = "normalization_std_CORRECT.npy"
FEAT_FILE    = "feature_list_CORRECT.txt"

BEST_MODEL_CALLBACK_PATH = f"best_model_{PREFIX}.keras"
CHECKPOINT_PATH          = f"lstm_model_{PREFIX}.keras"
FINAL_MODEL_PATH         = f"lstm_model_{PREFIX}_FINAL.keras"
TRAINING_PLOT_PATH       = f"training_{PREFIX}.png"
BEST_HYPERPARAMS_JSON    = f"best_hyperparams_{PREFIX}.json"
MODEL_CONFIG_JSON        = f"model_config_{PREFIX}.json"
PREDICTIONS_FILE         = f"predictions_LSTM_{PREFIX}.npy"
OPTIMAL_THRESHOLD_FILE   = f"optimal_threshold_{PREFIX}.npy"

# ATTENTION
@tf.keras.utils.register_keras_serializable()
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        e = tf.reduce_sum(e, axis=-1, keepdims=True)
        a = tf.nn.softmax(e, axis=1)
        return tf.reduce_sum(x * a, axis=1)

    def get_config(self):
        return super().get_config()

# CALLBACK
class DirectionalMetricsCallback(Callback):
    def __init__(self, val_data, best_model_path, target_accuracy=0.69, predict_batch_size=2048):
        super().__init__()
        self.Xv, self.Yv = val_data
        self.best_accuracy = 0.0
        self.history = {"accuracy": [], "precision": [], "recall": [], "f1": []}
        self.best_model_path = best_model_path
        self.target_accuracy = target_accuracy
        self.predict_batch_size = int(predict_batch_size)

    def on_epoch_end(self, epoch, logs=None):
        p = self.model.predict(self.Xv, batch_size=self.predict_batch_size, verbose=0).flatten()
        y = (p >= 0.5).astype(int)

        acc = accuracy_score(self.Yv, y)
        pr  = precision_score(self.Yv, y, zero_division=0)
        rc  = recall_score(self.Yv, y, zero_division=0)
        f1  = f1_score(self.Yv, y, zero_division=0)

        self.history["accuracy"].append(acc)
        self.history["precision"].append(pr)
        self.history["recall"].append(rc)
        self.history["f1"].append(f1)

        saved = ""
        if acc > self.best_accuracy:
            self.best_accuracy = acc
            self.model.save(self.best_model_path)
            saved = "SAVED"

        ok = "OK" if acc >= self.target_accuracy else "DANGER"
        print(f"\n   Acc:{acc:.3f}{ok} P:{pr:.3f} R:{rc:.3f} F1:{f1:.3f} {saved}")

# MODEL
def build_directional_model(timesteps, features, cnn_filters=64, lstm_units_1=128, lstm_units_2=64, dropout=0.3, name="MarketHeartbeat"):
    inp = layers.Input((timesteps, features))

    x = layers.Conv1D(cnn_filters, 3, activation="relu", padding="same")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv1D(cnn_filters, 3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units_1, return_sequences=True, dropout=dropout))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(lstm_units_2, return_sequences=True, dropout=dropout))(x)
    x = layers.BatchNormalization()(x)

    x = AttentionLayer()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation="sigmoid", name="direction")(x)
    return Model(inp, out, name=name)

# PLOT
def plot_training(hist, cb, save_path):
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    ax[0,0].plot(hist.history["loss"], label="Train", linewidth=2)
    ax[0,0].plot(hist.history["val_loss"], label="Val", linewidth=2)
    ax[0,0].set_title("Loss", fontweight="bold")
    ax[0,0].legend(); ax[0,0].grid(True, alpha=0.3)

    ax[0,1].plot(hist.history["auc"], label="Train", linewidth=2)
    ax[0,1].plot(hist.history["val_auc"], label="Val", linewidth=2)
    ax[0,1].set_title("AUC", fontweight="bold")
    ax[0,1].legend(); ax[0,1].grid(True, alpha=0.3)

    ax[0,2].plot(hist.history["accuracy"], label="Train", linewidth=2)
    ax[0,2].plot(hist.history["val_accuracy"], label="Val", linewidth=2)
    ax[0,2].axhline(TARGET_ACCURACY, color="red", linestyle="--", label=f"Target {TARGET_ACCURACY}")
    ax[0,2].set_title("Accuracy (Built-in)", fontweight="bold")
    ax[0,2].legend(); ax[0,2].grid(True, alpha=0.3)

    ax[1,0].plot(cb.history["accuracy"], linewidth=2)
    ax[1,0].axhline(TARGET_ACCURACY, color="red", linestyle="--", label="Profit Threshold")
    ax[1,0].axhline(0.5, color="gray", linestyle=":", label="Random")
    ax[1,0].set_title("Val Accuracy (Callback)", fontweight="bold")
    ax[1,0].legend(); ax[1,0].grid(True, alpha=0.3)

    ax[1,1].plot(cb.history["precision"], linewidth=2, label="Precision")
    ax[1,1].plot(cb.history["recall"], linewidth=2, label="Recall")
    ax[1,1].set_title("Precision & Recall", fontweight="bold")
    ax[1,1].legend(); ax[1,1].grid(True, alpha=0.3)

    ax[1,2].plot(cb.history["f1"], linewidth=2)
    ax[1,2].set_title("F1-Score", fontweight="bold")
    ax[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(" Plot:", save_path)
    plt.close()

# EVAL
def evaluate_split(model, name, X, Y, predict_batch_size=2048):
    print(f"\n--- {name} ---")
    p = model.predict(X, batch_size=int(predict_batch_size), verbose=0).flatten()
    y = (p >= 0.5).astype(int)

    acc = accuracy_score(Y, y)
    auc = roc_auc_score(Y, p)
    pr  = precision_score(Y, y, zero_division=0)
    rc  = recall_score(Y, y, zero_division=0)
    f1  = f1_score(Y, y, zero_division=0)

    print(f"   Accuracy: {acc:.4f} {'PROFITABLE!' if acc >= TARGET_ACCURACY else 'Below target'}")
    print(f"   AUC:      {auc:.4f}")
    print(f"   Precision:{pr:.4f}")
    print(f"   Recall:   {rc:.4f}")
    print(f"   F1:       {f1:.4f}")

    cm = confusion_matrix(Y, y)
    print("   Confusion Matrix:")
    print(f"      TN={cm[0,0]:,} FP={cm[0,1]:,}")
    print(f"      FN={cm[1,0]:,} TP={cm[1,1]:,}")

    print(f"\n   Classification Report ({name}):")
    print(classification_report(Y, y, target_names=["DOWN", "UP"], digits=4))
    return acc, auc, pr, rc, f1

# UTILS
def load_feature_names(path):
    feats = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                feats.append(parts[1].strip())
    return feats

def make_sequences(Xn, Y, seq_len):
    n = len(Xn)
    if n < seq_len:
        return np.empty((0, seq_len, Xn.shape[1]), np.float32), np.empty((0,), np.float32)

    try:
        from numpy.lib.stride_tricks import sliding_window_view
        Xs = sliding_window_view(Xn, window_shape=(seq_len, Xn.shape[1]))
        Xs = Xs[:, 0, :, :].astype(np.float32, copy=False)
        Ys = Y[seq_len-1:].astype(np.float32, copy=False)
        return Xs, Ys
    except Exception:
        Xs, Ys = [], []
        limit = n - seq_len + 1
        for i in range(limit):
            Xs.append(Xn[i:i+seq_len])
            Ys.append(Y[i+seq_len-1])
        return np.array(Xs, np.float32), np.array(Ys, np.float32)

# MAIN
if __name__ == "__main__":
    print("="*80)
    print("MARKET HEARTBEAT - TRAINER (H10) - CORRECT SPLIT")
    print("Train: TRAIN parquet + X_train_CORRECT.npy")
    print("Search: OPTIMIZE parquet")
    print("Final OOS: TEST parquet (Q4 2024)")
    print("="*80)

    needed = [PARQ_TRAIN, PARQ_OPT, PARQ_TEST, X_TRAIN_FILE, Y_TRAIN_FILE, MEAN_FILE, STD_FILE, FEAT_FILE]
    miss = [f for f in needed if not os.path.exists(f)]
    if miss:
        print("\n Missing files:")
        for f in miss:
            print(" -", f)
        raise SystemExit(1)

    FEATURES = load_feature_names(FEAT_FILE)
    MEAN = np.load(MEAN_FILE)
    STD  = np.load(STD_FILE)

    X_train = np.load(X_TRAIN_FILE)
    Y_train = np.load(Y_TRAIN_FILE)

    timesteps = X_train.shape[1]
    n_features = X_train.shape[2]

    print(f"\n TRAIN seq: X={X_train.shape}, Y={Y_train.shape}")
    print(f"   timesteps={timesteps}, features={n_features}, horizon=H{H} (~{H*30/60:.1f} min)")

    cw_arr = compute_class_weight("balanced", classes=np.unique(Y_train), y=Y_train)
    class_weights = {0: cw_arr[0], 1: cw_arr[1]}
    print(f"\n️ Class weights (TRAIN): DOWN={class_weights[0]:.3f}, UP={class_weights[1]:.3f}")

    def load_norm_seq_from_parquet(parq_path, name):
        df = pd.read_parquet(parq_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        X = df[FEATURES].values
        Y = df["Target_Y"].values
        Xn = (X - MEAN) / (STD + 1e-8)
        Xs, Ys = make_sequences(Xn, Y, SEQ_LEN)
        print(f" {name} seq: X={Xs.shape}, Y={Ys.shape}")
        return Xs, Ys

    X_opt, Y_opt = load_norm_seq_from_parquet(PARQ_OPT, "OPTIMIZE")
    X_test, Y_test = load_norm_seq_from_parquet(PARQ_TEST, "TEST(OOS)")

    Y_all = np.concatenate([Y_train, Y_opt, Y_test], axis=0)
    up = int((Y_all == 1).sum()); dn = int((Y_all == 0).sum())
    print("\n Target Distribution (ALL):")
    print(f"   UP:   {up:,} ({up/len(Y_all)*100:.2f}%)")
    print(f"   DOWN: {dn:,} ({dn/len(Y_all)*100:.2f}%)")
    print(f"   Balance Ratio: {min(up,dn)/max(up,dn):.2%}")

    print("\n" + "="*80)
    print(f" RANDOM SEARCH ({PREFIX})  Train=TRAIN  Score=OPTIMIZE")
    print("="*80)

    best_conf = None
    best_val_auc = -1.0

    for t in range(N_TRIALS):
        conf = {k: random.choice(v) for k, v in HYPER.items()}
        print(f"\n--- Trial {t+1}/{N_TRIALS} ---")
        print("Config:", conf)

        model = build_directional_model(
            timesteps, n_features,
            cnn_filters=conf["cnn_filters"],
            lstm_units_1=conf["lstm_units_1"],
            lstm_units_2=conf["lstm_units_2"],
            dropout=conf["dropout"],
            name=f"MarketHeartbeat_{PREFIX}_RS"
        )

        model.compile(
            optimizer=Adam(conf["initial_lr"]),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")]
        )

        es = EarlyStopping(monitor="val_auc", patience=5, restore_best_weights=True, mode="max", verbose=0)
        rl = ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=3, min_lr=1e-6, verbose=0)

        hist = model.fit(
            X_train, Y_train,
            validation_data=(X_opt, Y_opt),
            epochs=SEARCH_EPOCHS,
            batch_size=conf["batch_size"],
            class_weight=class_weights,
            callbacks=[es, rl],
            verbose=0
        )

        val_auc = float(max(hist.history["val_auc"]))
        print(f"   → Best val_auc = {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_conf = conf

    print("\n BEST CONFIG:")
    print(best_conf)
    print(f"Best val_auc: {best_val_auc:.4f}")

    with open(BEST_HYPERPARAMS_JSON, "w") as f:
        json.dump(
            {
                "best_conf": best_conf,
                "best_val_auc": best_val_auc,
                "horizon": f"H{H}",
                "strategy": "Random search: fit TRAIN, select on OPTIMIZE. TEST used only at the end."
            },
            f, indent=2
        )

    print("\n" + "="*80)
    print(f" FINAL TRAINING ({PREFIX})  Train=TRAIN  Monitor=OPTIMIZE")
    print("TEST is only for final evaluation (OOS).")
    print("="*80)

    model = build_directional_model(
        timesteps, n_features,
        cnn_filters=best_conf["cnn_filters"],
        lstm_units_1=best_conf["lstm_units_1"],
        lstm_units_2=best_conf["lstm_units_2"],
        dropout=best_conf["dropout"],
        name=f"MarketHeartbeat_{PREFIX}_FINAL"
    )

    model.compile(
        optimizer=Adam(best_conf["initial_lr"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ]
    )

    print(f" Params: {model.count_params():,}")
    print(f" Target Accuracy ≥ {TARGET_ACCURACY:.2f}")

    cb_dir = DirectionalMetricsCallback(
        (X_opt, Y_opt),
        BEST_MODEL_CALLBACK_PATH,
        TARGET_ACCURACY,
        predict_batch_size=2048
    )
    es2 = EarlyStopping(monitor="val_auc", patience=10, restore_best_weights=True, mode="max", verbose=1)
    rl2 = ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ck  = ModelCheckpoint(CHECKPOINT_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=0)

    hist = model.fit(
        X_train, Y_train,
        validation_data=(X_opt, Y_opt),
        epochs=EPOCHS,
        batch_size=best_conf["batch_size"],
        class_weight=class_weights,
        callbacks=[cb_dir, es2, rl2, ck],
        verbose=1
    )

    plot_training(hist, cb_dir, TRAINING_PLOT_PATH)

    try:
        model = keras.models.load_model(BEST_MODEL_CALLBACK_PATH, custom_objects={"AttentionLayer": AttentionLayer})
        print("\n Loaded best callback model:", BEST_MODEL_CALLBACK_PATH)
    except Exception:
        print("\n Could not load best callback model, using in-memory model.")

    print("\n" + "="*80)
    print(f" EVALUATION ({PREFIX})")
    print("="*80)

    val_metrics  = evaluate_split(model, "OPTIMIZE", X_opt, Y_opt, predict_batch_size=2048)
    test_metrics = evaluate_split(model, "TEST_OOS", X_test, Y_test, predict_batch_size=2048)

    acc_test = test_metrics[0]
    if acc_test > 0.55:
        edge = (acc_test - 0.5) * 2
        exp_ann = edge * 100 * 252
        print("\n TRADING POTENTIAL (from TEST OOS):")
        print(f"   Edge over random: {edge*100:.2f}%")
        print(f"   Expected annual return (rough): {exp_ann:.1f}%")

    print("\n Saving final artifacts...")

    model.save(FINAL_MODEL_PATH)

    X_full = np.concatenate([X_train, X_opt, X_test], axis=0)
    preds_full = model.predict(X_full, batch_size=2048, verbose=1).flatten()

    np.save(PREDICTIONS_FILE, preds_full)
    np.save(OPTIMAL_THRESHOLD_FILE, 0.5)

    with open(MODEL_CONFIG_JSON, "w") as f:
        json.dump(
            {
                "threshold": 0.5,
                "hyperparams": best_conf,
                "optimize_metrics": {
                    "accuracy": float(val_metrics[0]),
                    "auc": float(val_metrics[1]),
                    "precision": float(val_metrics[2]),
                    "recall": float(val_metrics[3]),
                    "f1": float(val_metrics[4]),
                },
                "test_metrics": {
                    "accuracy": float(test_metrics[0]),
                    "auc": float(test_metrics[1]),
                    "precision": float(test_metrics[2]),
                    "recall": float(test_metrics[3]),
                    "f1": float(test_metrics[4]),
                },
                "profitable_test": bool(test_metrics[0] >= TARGET_ACCURACY),
                "dataset_prefix": PREFIX,
                "horizon": {"bars": H, "minutes": H * 30.0 / 60.0},
                "split_strategy": "Correct splits: TRAIN (Jan23-Jun24), OPTIMIZE (Jul-Sep24), TEST/Q4 (Oct-Dec24).",
                "files_used": {
                    "train_seq": [X_TRAIN_FILE, Y_TRAIN_FILE],
                    "parquets": [PARQ_TRAIN, PARQ_OPT, PARQ_TEST],
                    "normalization": [MEAN_FILE, STD_FILE],
                    "feature_list": FEAT_FILE,
                },
            },
            f, indent=2
        )

    print("\n" + "="*80)
    print(f" TRAINING COMPLETE ({PREFIX})")
    print("Saved:")
    print(" -", BEST_MODEL_CALLBACK_PATH)
    print(" -", CHECKPOINT_PATH)
    print(" -", FINAL_MODEL_PATH)
    print(" -", TRAINING_PLOT_PATH)
    print(" -", BEST_HYPERPARAMS_JSON)
    print(" -", MODEL_CONFIG_JSON)
    print(" -", PREDICTIONS_FILE)
    print(" -", OPTIMAL_THRESHOLD_FILE)
    print("="*80)
