import os
import numpy as np
import joblib
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from config import LABELS, MODEL_DIR, CONF_THRESHOLD
from features import fix_length, ml_vector

BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, MODEL_DIR, "svm.joblib"))

app = FastAPI()

@app.get("/")
def index():
    return FileResponse(os.path.join(BASE, "index.html"))

@app.get("/info")
def info():
    return {"labels": LABELS, "threshold": CONF_THRESHOLD, "model": "MFCC + SVM"}

@app.post("/predict")
async def predict(req: Request):
    raw = await req.body()
    y = np.frombuffer(raw, dtype=np.float32).copy()
    if y.size < 1600:
        return JSONResponse({"error": "audio qua ngan"}, status_code=400)
    x = ml_vector(fix_length(y)).reshape(1, -1)
    p = model.predict_proba(x)[0]
    classes = [int(c) for c in model.classes_]
    i = int(np.argmax(p))
    return {"label": LABELS[classes[i]], "conf": float(p[i]),
            "probs": {LABELS[c]: float(v) for c, v in zip(classes, p)}}

if __name__ == "__main__":
    print("Mo trinh duyet: http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)