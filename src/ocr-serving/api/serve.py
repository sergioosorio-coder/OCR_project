from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import io
import numpy as np 

from ocr_pipeline.serving import OcrPipeline

MODEL_PATH = "models/trocr-line" #"microsoft/trocr-base-handwritten" 

app = FastAPI(title="Notebook OCR API")

ocr_pipeline = OcrPipeline(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    
    image_bgr = read_image_cv2(content)
    if image_bgr is None:
        raise ValueError("No se pudo decodificar la imagen con OpenCV")
    
    result = ocr_pipeline.predict_page(image_bgr)
    return JSONResponse(content=result)


def read_image_cv2(file_bytes: bytes):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
    return img