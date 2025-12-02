from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import cv2

from ocr_pipeline.serving import OcrPipeline
from ocr_pipeline.line_extractor import detect_lines

def save_crops(image, boxes, outdir="data/crops"):

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = image[y1:y2, x1:x2]
        outpath = outdir / f"crop_{i:02d}.png"
        cv2.imwrite(str(outpath), crop)
        print(f"Guardado: {outpath}")

def main():

    img_path = Path("data/IMG_7330.jpg") 
    image = cv2.imread(img_path)

    model_path = "models/trocr-line" # "microsoft/trocr-base-handwritten" 
    pipeline = OcrPipeline(model_path)
    boxes,_,_,_  = detect_lines(image)
    save_crops(image, boxes)
    
    result = pipeline.predict_page(image)

    print("\n===== RESULTADOS OCR =====")
    for i, line in enumerate(result["lines"]):
        print(f"[{i}] {line['text']}   box={line['box']}")

    print("\nTexto completo:\n")
    print(result["full_text"])


if __name__ == "__main__":
    main()
