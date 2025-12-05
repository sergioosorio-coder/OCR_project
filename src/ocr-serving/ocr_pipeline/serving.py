from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .line_extractor import detect_lines

Box = Tuple[int, int, int, int]

class OcrPipeline:
    def __init__(self, model_path: str, max_new_tokens: int = 128, num_beams: int = 4) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()

        self.model.config.decoder_start_token_id = self.processor.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.processor.tokenizer.eos_token_id
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

        self.generation_kwargs = dict(
            max_new_tokens=max_new_tokens, 
            num_beams=num_beams,
            early_stopping=True,
            length_penalty=1.1,            
            no_repeat_ngram_size=2,        
        )

    def _crops_from_boxes(self, page_image: np.ndarray, boxes: List[Box]):

        crops = []
        for (x1, y1, x2, y2) in boxes:
            crop = page_image[y1:y2, x1:x2]
            crops.append(
                Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            )
        return crops

    def _predict_lines(self, line_images: List[np.ndarray]) -> List[str]:
        encoded = self.processor(images=line_images, return_tensors="pt")
        pixel_values = encoded.pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                **self.generation_kwargs,
            )

        decoded = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip() for t in decoded]

    def predict_page(self, page_image: np.ndarray) -> Dict[str, Any]:

        boxes,_,_,_ = detect_lines(page_image)
        if not boxes:
            return {"lines": [], "full_text": ""}

        crops = self._crops_from_boxes(page_image, boxes)
        crops_preprocessed = [preprocess_line_bgr(np.array(crop)) for crop in crops]
        texts = self._predict_lines(crops_preprocessed)
        full_text = "\n".join(texts)

        return {
            "lines": [{"box": box, "text": text} for box, text in zip(boxes, texts)],
            "full_text": full_text,
        }
    
def preprocess_line_bgr(line_bgr: np.ndarray, target_height: int = 384) -> Image.Image:
    """
    Preprocesa el recorte de una línea:
    - pasa de BGR (cv2) a RGB
    - normaliza tamaño manteniendo aspect ratio (altura target_height)
    - mejora un poco el contraste si la imagen está muy "lavada"
    - devuelve un PIL.Image listo para el TrOCRProcessor
    """
    # 1) BGR -> RGB
    line_rgb = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2RGB)

    # 2) Resize manteniendo aspect ratio
    h, w, _ = line_rgb.shape
    scale = target_height / float(h)
    new_w = int(w * scale)
    line_resized = cv2.resize(line_rgb, (new_w, target_height), interpolation=cv2.INTER_LINEAR)

    # 3) Pequeño ajuste de contraste usando CLAHE sobre luma
    #    (opcional, pero suele ayudar en cuadernos con mala iluminación)
    lab = cv2.cvtColor(line_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    line_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # 4) Convertir a PIL
    pil_img = Image.fromarray(line_enhanced)

    return pil_img