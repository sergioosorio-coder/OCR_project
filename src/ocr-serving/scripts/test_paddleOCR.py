import cv2
import json
import numpy as np
from pathlib import Path
from paddleocr import TextDetection, TextRecognition
import matplotlib.pyplot as plt
from typing import Tuple

def load_image(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Path Error: {image_path}")
    return img

def init_ocr(detector_model_name: str = "PP-OCRv5_server_det",
             recognizer_model_name: str =  "PP-OCRv5_server_rec",
             device: str = "cpu") -> Tuple[TextDetection,TextRecognition]:
    
    ocr_det = TextDetection(
        model_name = detector_model_name,
        device = device,
    )

    ocr_rec = TextRecognition(
        model_name = recognizer_model_name,
        device = device,
    )

    return ocr_det,ocr_rec



def detect(ocr_det: TextDetection, image_path: str):

    result = ocr_det.predict(image_path, batch_size = 1)
    
    all_polys = [] 
    all_scores = [] 
    for res in result:
        data = res.json["res"]
        dt_polys = data["dt_polys"]  
        dt_scores = data["dt_scores"]

        for poly, score in zip(dt_polys, dt_scores):
            all_polys.append(poly)
            all_scores.append(score)
            print("bbox:", poly, "score:", float(score))
    
    img = load_image(image_path)
    bboxes_xyxy = []
    for poly in all_polys:
        xs = np.array([coord[0] for coord in poly])
        ys = np.array([coord[1] for coord in poly])
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        bboxes_xyxy.append((x_min, y_min, x_max, y_max))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0),2)
    
    return bboxes_xyxy

def recognize(ocr_rec: TextRecognition, image_path, bboxes, out_path):
    
    img = load_image(image_path)
    crops = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        crop = img[y_min:y_max,x_min:x_max]
        crops.append(crop)

    rec_outputs = ocr_rec.predict(crops, batch_size=8)
    vis = img.copy()
    for (x1, y1, x2, y2), rec in zip(bboxes, rec_outputs):
        text = rec.get('rec_text')
        score = rec.get('rec_score')
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{text} ({score:.2f})"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, vis)
    print(f"Imagen con bboxes + texto guardada en: {out_path}")
    return rec_outputs

def main(image_path, out_path: str = "./data/paddleOCR/salida_det.png"):

    if not Path(image_path).exists():
        raise SystemExit(f"La imagen {image_path} no existe.")

    ocr_det, ocr_rec = init_ocr() 
    bboxes = detect(ocr_det, image_path)
    rec_outputs = recognize(ocr_rec,image_path, bboxes,out_path)

if __name__ == "__main__":
#IMG_7330
#en_hw2022_01_1
#en_hw2022_07_IMG_20211001_135020
    path = 'data/IMG_7330.jpg'
    main(path)