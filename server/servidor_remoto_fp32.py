r"""
Microservicio API REST — Paradigma B (Procesamiento Remoto en Servidor FP32)
Tesis: Evaluación del desempeño en la detección de LLA (C-NMC 2019)
Autor: José Manuel Mariaca Patiño (USIL)

Ejecución:
  C:\Users\gunte\.lla_env\Scripts\python.exe servidor_remoto_fp32.py
Endpoint:
  POST http://<IP_LOCAL>:8000/predict
"""
import os
import time
try:
    import psutil
except ImportError:
    psutil = None
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_lla_fp32.keras")
IMG_SIZE = 224

app = FastAPI(
    title="API REST Detección de LLA — Paradigma B (EfficientNet-B0 FP32)",
    version="1.0.0",
)

model = None


def preprocess_image_bytes(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("No se pudo decodificar la imagen enviada.")

    # 1. CLAHE en canal L de LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    lab = cv2.merge([l_ch, a_ch, b_ch])
    normalized_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2. Suavizado Gaussiano 3x3
    blurred = cv2.GaussianBlur(normalized_bgr, (3, 3), 0)

    # 3. Segmentación celular completa (núcleo + citoplasma) y recorte ROI con 20% de margen
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 15, 20], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        pad_x, pad_y = int(w * 0.20), int(h * 0.20)
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(blurred.shape[1], x + w + pad_x), min(blurred.shape[0], y + h + pad_y)
        roi = blurred[y1:y2, x1:x2]
        if roi.size == 0:
            roi = blurred
    else:
        roi = blurred

    resized_bgr = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    return resized_rgb.astype(np.float32)[np.newaxis, ...]


@app.on_event("startup")
def load_fp32_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = keras.models.load_model(MODEL_PATH)
        # Warm-up inicial
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _ = model.predict(dummy, verbose=0)
        print(f"[FastAPI FP32] Modelo cargado exitosamente desde: {MODEL_PATH}")
    else:
        print(f"[FastAPI FP32] ADVERTENCIA: Aún no existe {MODEL_PATH}")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None, "paradigm": "Remote_FP32"}


@app.post("/predict")
async def predict_cell(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo FP32 no cargado.")
    t0 = time.perf_counter()
    raw_bytes = await file.read()
    try:
        tensor_in = preprocess_image_bytes(raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    probs = model.predict(tensor_in, verbose=0)[0]
    latency_ms = (time.perf_counter() - t0) * 1000.0
    prob_all = float(probs[0])
    prob_hem = float(probs[1])
    label = "ALL" if prob_all >= prob_hem else "HEM"
    ram_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0) if psutil is not None else 0.0

    return {
        "paradigm": "Remote_FP32",
        "label": label,
        "allProbability": prob_all,
        "hemProbability": prob_hem,
        "serverLatencyMs": round(latency_ms, 2),
        "serverRamMb": round(ram_mb, 2),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
