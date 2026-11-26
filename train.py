import os, glob, numpy as np, joblib
from config import LABELS, DATA_DIR, MODEL_DIR, MAX_PER_CLASS
from features import load_wav, ml_vector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def speaker_of(path):
    return os.path.basename(path).split("_")[0]

def load_dataset():
    X, ys, groups = [], [], []
    for i, lab in enumerate(LABELS):
        files = sorted(glob.glob(os.path.join(DATA_DIR, lab, "*.wav")))
        if MAX_PER_CLASS:
            files = files[:MAX_PER_CLASS]
        print(f"{lab}: {len(files)} file")
        for k, f in enumerate(files):
            X.append(ml_vector(load_wav(f)))
            ys.append(i)
            groups.append(speaker_of(f))
            if (k + 1) % 500 == 0:
                print(f"  da xu ly {k + 1}")
    return np.array(X), np.array(ys), np.array(groups)

def group_split(idx, ys, groups, test_size, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(gss.split(idx, ys[idx], groups[idx]))
    return idx[a], idx[b]

def report(name, y_true, y_pred):
    print(f"\n=== {name} (tap test) === accuracy = {accuracy_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
    print(confusion_matrix(y_true, y_pred))

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, ys, groups = load_dataset()
    if len(X) == 0:
        raise SystemExit("Chua co data. Chay truoc: py -3.14 download_data.py")

    idx = np.arange(len(ys))
    tr, rest = group_split(idx, ys, groups, 0.30)
    va, te = group_split(rest, ys, groups, 0.50)
    print(f"train {len(tr)} | val {len(va)} | test {len(te)} | so dac trung {X.shape[1]}")

    # 1) Chon tham so C cho SVM bang tap validation
    best_c, best_acc = 1, -1.0
    for c in [1, 3, 10, 30]:
        m = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma="scale"))
        m.fit(X[tr], ys[tr])
        acc = accuracy_score(ys[va], m.predict(X[va]))
        print(f"SVM C={c}: val accuracy = {acc:.3f}")
        if acc > best_acc:
            best_c, best_acc = c, acc
    print(f"Chon C = {best_c}")

    # 2) Train cac mo hinh tren train + val, danh gia tren test
    trva = np.concatenate([tr, va])
    models = [
        ("SVM", make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(kernel="rbf", C=best_c, gamma="scale"), ensemble=False, cv=3))),
    ]
    for name, clf in models:
        clf.fit(X[trva], ys[trva])
        report(name, ys[te], clf.predict(X[te]))
        joblib.dump(clf, os.path.join(MODEL_DIR, f"{name.lower()}.joblib"))

    print("\nXong! Web dung models/svm.joblib. Chay: run_web.bat")

if __name__ == "__main__":
    main()