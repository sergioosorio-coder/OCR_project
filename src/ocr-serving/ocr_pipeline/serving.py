from typing import List, Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from .line_extractor import detect_lines

Box = Tuple[int, int, int, int]

class OcrPipeline:
    def __init__(self, model_path: str, max_new_tokens: int = 128) -> None:
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.max_new_tokens = max_new_tokens

    def _crops_from_boxes(self, page_image: np.ndarray, boxes: List[Box]):

        img_cv = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
        crops = []
        for (x1, y1, x2, y2) in boxes:
            crop = img_cv[y1:y2, x1:x2]
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
                max_new_tokens=self.max_new_tokens,
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
        texts = self._predict_lines(crops)
        full_text = "\n".join(texts)

        return {
            "lines": [{"box": box, "text": text} for box, text in zip(boxes, texts)],
            "full_text": full_text,
        }