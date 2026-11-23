import os, glob, numpy as np, joblib
from config import LABELS, DATA_DIR, MODEL_DIR, MAX_PER_CLASS
from features import load_wav, mfcc_sequence, mfcc_vector
from tensorflow.keras import layers, models, callbacks
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

def speaker_of(path):
    return os.path.basename(path).split("_")[0]

def load_dataset():
    seqs, ys, groups = [], [], []
    for i, lab in enumerate(LABELS):
        files = sorted(glob.glob(os.path.join(DATA_DIR, lab, "*.wav")))
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]
        print(f"{lab}: {len(files)} file")
        for k, f in enumerate(files):
            seqs.append(mfcc_sequence(load_wav(f)))
            ys.append(i)
            groups.append(speaker_of(f))
            if (k + 1) % 500 == 0:
                print(f"  da xu ly {k + 1}")
    return seqs, np.array(ys), np.array(groups)

def group_split(idx, ys, groups, test_size, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(gss.split(idx, ys[idx], groups[idx]))
    return idx[a], idx[b]

def build_lstm(T, F, n_cls):
    m = models.Sequential([
        layers.Input((T, F)),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(n_cls, activation="softmax"),
    ])
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m

def main(epochs=30):
    os.makedirs(MODEL_DIR, exist_ok=True)
    seqs, ys, groups = load_dataset()
    if len(seqs) == 0:
        raise SystemExit("Chua co data. Chay truoc: py -3.14 download_data.py")

    T = max(s.shape[0] for s in seqs)
    F = seqs[0].shape[1]
    X = np.zeros((len(seqs), T, F), np.float32)
    for i, s in enumerate(seqs):
        X[i, :s.shape[0]] = s

    idx = np.arange(len(ys))
    tr, rest = group_split(idx, ys, groups, 0.30)
    va, te = group_split(rest, ys, groups, 0.50)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} (chia theo nguoi noi)")

    flat = X[tr].reshape(-1, F)
    mean, std = flat.mean(0), flat.std(0) + 1e-6
    Xn = (X - mean) / std

    model = build_lstm(T, F, len(LABELS))
    model.summary()
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(Xn[tr], ys[tr], validation_data=(Xn[va], ys[va]),
              epochs=epochs, batch_size=64, callbacks=[es], verbose=2)
    pred = model.predict(Xn[te], verbose=0).argmax(1)
    print("\n=== LSTM (tap test) ===")
    print(classification_report(ys[te], pred, target_names=LABELS, digits=3))
    print(confusion_matrix(ys[te], pred))
    model.save(os.path.join(MODEL_DIR, "lstm.keras"))
    np.savez(os.path.join(MODEL_DIR, "norm.npz"), mean=mean, std=std, T=T)

    V = np.stack([mfcc_vector(s) for s in seqs])
    for name, clf in [
        ("SVM", make_pipeline(StandardScaler(), SVC(kernel="rbf"))),
        ("RandomForest", make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, random_state=0))),
    ]:
        clf.fit(V[tr], ys[tr])
        p = clf.predict(V[te])
        print(f"\n=== {name} (tap test) ===")
        print(classification_report(ys[te], p, target_names=LABELS, digits=3))
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.lower()}.joblib"))

    print("\nXong! Tiep theo: py -3.14 demo.py")

if __name__ == "__main__":
    main()