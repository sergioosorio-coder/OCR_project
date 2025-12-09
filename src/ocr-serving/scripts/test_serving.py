import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2

from ocr_pipeline.serving import OcrPipeline
from ocr_pipeline.line_extractor import detect_lines
from ocr_pipeline.word_extractor import detect_words, extract_word_crops


def save_crops(image, boxes, outdir="data/crops"): 
    outdir = Path(outdir) 
    outdir.mkdir(parents=True, exist_ok=True) 
    image_crops = [] 
    for i, (x1, y1, x2, y2) in enumerate(boxes): 
        crop = image[y1:y2, x1:x2] 
        outpath = outdir / f"crop_{i:02d}.png" 
        cv2.imwrite(str(outpath), crop) 
        print(f"Guardado: {outpath}") 
        image_crops.append(crop) 
    return image_crops

def main(path):
    img_path = Path(path) 
    image = cv2.imread(img_path)

    model_path = "models/trocr-words" # "microsoft/trocr-base-handwritten" # 
    pipeline = OcrPipeline(model_path)
    line_boxes,_,_,_  = detect_lines(image)

    full_text = []
    crops_dir = Path("data/crops")
    crops_dir.mkdir(exist_ok=True, parents=True)

    for li, (x1, y1, x2, y2) in enumerate(line_boxes):
        print(f"\n=== Procesando línea {li} ===")
        line_crop = image[y1:y2, x1:x2]
        cv2.imwrite(str(crops_dir / f"line_{li:02d}.png"), line_crop)
        word_boxes = detect_words(line_crop)
        word_crops = extract_word_crops(line_crop, word_boxes)
        line_predictions = []

        for wi, word_bgr in enumerate(word_crops):
            rgb = cv2.cvtColor(word_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            try:
                preds = pipeline.predict_text(pil) 
                text = "\n".join(preds)
            except Exception as e:
                print(f"Error OCR en word {wi}: {e}")
                text = ""
            
            line_predictions.append(text)
            cv2.imwrite(
                str(crops_dir / f"line_{li:02d}_word_{wi:02d}.png"),
                word_bgr
            )
        line_text = " ".join(line_predictions)
        print(f"Línea reconocida: {line_text}")
        full_text.append(line_text)

    print("\n\n====== TEXTO COMPLETO ======")
    print("\n".join(full_text))

if __name__ == "__main__":
# "data/IMG_7330.jpg"
# "data/en_hw2022_01_1.jpg"
# "data/en_hw2022_07_IMG_20211001_135020.jpg"
    path = "data/IMG_7330.jpg"
    main(path)






