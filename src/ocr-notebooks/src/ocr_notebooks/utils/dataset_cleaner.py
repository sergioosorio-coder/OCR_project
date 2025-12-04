from typing import List, Tuple, Dict, Any
from PIL import Image
import numpy as np
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from evaluate import load

cer_metric = load("cer")

def _load_model(model_dir: str, device: str | None = None):
    if device is None:
        device = "mps" if torch.mps.is_available() else "cpu"
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)
    model.eval()
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    return processor, model, device

@torch.inference_mode()
def _predict_line(
    image: Any,
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    device: str,
) -> str:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")

    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    ids = model.generate(pixel_values, max_new_tokens=64, num_beams=4, early_stopping=True)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def clean_lines_samples(
    samples: List[Dict[str, Any]],
    model_dir: str,
    cer_threshold: float = 0.25,
    max_samples: int | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float]]:

    if max_samples is not None:
        samples = samples[:max_samples]

    processor, model, device = _load_model(model_dir)
    clean, bad = [], []

    for s in samples:
        gt = s.get("text", "") or ""
        pred = _predict_line(s["image"], processor, model, device)
        cer = cer_metric.compute(predictions=[pred], references=[gt])
        out = {**s, "cer_base": float(cer)}
        (clean if cer <= cer_threshold else bad).append(out)

    total = len(samples) or 1
    stats = {
        "total": float(total),
        "kept": float(len(clean)),
        "dropped": float(len(bad)),
        "kept_ratio": float(len(clean) / total),
    }
    return clean, bad, stats