import os
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from config import LABELS, MODEL_DIR, CONF_THRESHOLD
from features import fix_length, mfcc_sequence

BASE = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE, MODEL_DIR, "lstm.keras"))
nz = np.load(os.path.join(BASE, MODEL_DIR, "norm.npz"))
MEAN, STD, T = nz["mean"], nz["std"], int(nz["T"])

app = FastAPI()

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "index.html"))

@app.get("/info")
def info():
    return {"labels": LABELS, "threshold": CONF_THRESHOLD, "model": "MFCC + LSTM"}

@app.post("/predict")
async def predict(req: Request):
    raw = await req.body()
    y = np.frombuffer(raw, dtype=np.float32).copy()
    if y.size < 1600:
        return JSONResponse({"error": "audio qua ngan"}, status_code=400)
    seq = mfcc_sequence(fix_length(y))
    X = np.zeros((1, T, seq.shape[1]), np.float32)
    n = min(seq.shape[0], T)
    X[0, :n] = seq[:n]
    X = (X - MEAN) / STD
    p = model(X, training=False).numpy()[0]
    i = int(p.argmax())
    return {"label": LABELS[i], "conf": float(p[i]),
            "probs": {lab: float(v) for lab, v in zip(LABELS, p)}}

if __name__ == "__main__":
    print("Mo trinh duyet: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)