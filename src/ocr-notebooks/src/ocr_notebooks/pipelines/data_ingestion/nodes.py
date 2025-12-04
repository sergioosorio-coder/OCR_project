"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 1.1.1
"""

from huggingface_hub import hf_hub_download
import shutil
from pathlib import Path
# from __future__ import annotations
from typing import Dict, Any, List
from io import BytesIO
from zipfile import ZipFile

import re
import cv2 
import numpy as np 

from PIL import Image
import unicodedata
from ocr_notebooks.utils.dataset_cleaner import clean_lines_samples

def download_files_from_hf(repo_id: str, files: dict, output_dir: str) -> dict:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    output_paths = {}

    for key, filename in files.items():
        local_path = hf_hub_download(repo_id, repo_type="dataset", filename=filename)
        dest_path = Path(output_dir) / filename
        shutil.copy(local_path, dest_path)
        output_paths[key] = str(dest_path)

    return output_paths

def _get_text_from_ann(ann):
    attrs = ann.get("attributes", {})
    trans = attrs.get("translation", "")
    if isinstance(trans, dict):
        return trans.get("en") or (list(trans.values())[0] if trans else "")
    return trans or ""

def _get_image_coords(ann):
    segmentation = ann.get("segmentation",[])
    coords = segmentation[0] if isinstance(segmentation[0], (list, tuple)) else segmentation
    return coords

def _get_image_size(ann, image_name_dict):
    image_id = ann['image_id']
    img_info = image_name_dict[image_id]
    return img_info['width'], img_info['height']

def _polygon_to_bbox(coords):
    xs = np.array(coords[0::2])
    ys = np.array(coords[1::2])
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return int(x_min), int(y_min), int(x_max), int(y_max)

def _expand_bbox(bbox, img_w, img_h, margin=4):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img_w, x2 + margin)
    y2 = min(img_h, y2 + margin)
    return x1, y1, x2, y2

def _get_text_from_ann(ann):
    attrs = ann.get("attributes", {})
    trans = attrs.get("translation", "")
    if isinstance(trans, dict):
        return trans.get("en") or (list(trans.values())[0] if trans else "")
    return trans or ""

def cy(word):
    x1, y1, x2, y2 = word["bbox"]
    return 0.5 * (y1 + y2)

def h(word):
    x1, y1, x2, y2 = word["bbox"]
    return (y2 - y1)

def group_words_into_lines(
    words: list,
    max_center_diff_factor: float = 0.7,
)-> List[dict]:

    if not words:
        return []

    # Ordenamos primero por cy y luego por x para tener algo estable
    words = sorted(words, key=lambda w: (cy(w), w["bbox"][0]))

    lines: List[dict] = []

    for w in words:
        placed = False
        w_cy = cy(w)
        w_h  = h(w)
        for line in lines:
            cy_mean = line["cy_mean"]
            h_mean  = line["h_mean"]
            if abs(w_cy - cy_mean) <= max_center_diff_factor * h_mean:
                line["words"].append(w)

                n = len(line["words"])
                line["cy_mean"] = (cy_mean * (n - 1) + w_cy) / n
                line["h_mean"]  = (h_mean * (n - 1) + w_h)  / n
                placed = True
                break
        if not placed:
            
            lines.append({
                "words": [w],
                "cy_mean": w_cy,
                "h_mean":  w_h,
            })
    output_lines = []
    for line in lines:
        sorted_line = sorted(line["words"], key=lambda w: w["bbox"][0])
        output_lines.append(sorted_line)

    return output_lines


def merge_line_words(
    line_words: list,
    v_pad: int = 4,
    expand_factor: float = 1.2,
):
    """
    Construye la caja (x_min, y_min, x_max, y_max) de una línea,
    robusto frente a palabras con bboxes “contaminados” verticalmente.
    """

    if not line_words:
        raise ValueError("line_words no puede ser vacío")

    xs1 = [w["bbox"][0] for w in line_words]
    xs2 = [w["bbox"][2] for w in line_words]

    x_min = min(xs1)
    x_max = max(xs2)

    cys = np.array([cy(w) for w in line_words], dtype=float)
    hs  = np.array([h(w)  for w in line_words], dtype=float)

    cy_line = float(np.mean(cys))
    h_med   = float(np.median(hs))  

    half_height = 0.5 * h_med * expand_factor

    y_min = int(cy_line - half_height) - v_pad
    y_max = int(cy_line + half_height) + v_pad

    return x_min, max(0, y_min), x_max, y_max


def extract_samples(
        coco_json_file:Dict[str, Any],
        images_zip_data: str|Path|bytes,
        n_pages: int = 20,
        model_dir: str = "microsoft/trocr-base-handwritten",
        cer_threshold: float = 0.25,
                    ):
    categories_dict = {cat['id']:cat['name'] for cat in coco_json_file['categories']}
    image_name_dict = {image['id']:{'file_name':image['file_name'],
                                'height':image['height'],
                                'width':image['width'] 
                                } for image in coco_json_file['images']}
    annotations = coco_json_file['annotations']

    if isinstance(images_zip_data, (str, Path)):
        zf = ZipFile(images_zip_data, "r")
    else:
        zf = ZipFile(BytesIO(images_zip_data))

    name_map: Dict[str, str] = {}
    for member in zf.namelist():
        if member.endswith("/"):
            continue  # es carpeta
        basename = Path(member).name
        name_map[basename] = member

    line_samples= []
    for img_id in range(min(n_pages,len(image_name_dict))):
        images_name = image_name_dict[img_id]['file_name']
        zip_member = name_map.get(images_name,'Empty')
        with zf.open(zip_member) as f:
            pil_img = Image.open(f).convert("RGB")
        annotations_by_page = [ann for ann in annotations if ann.get('image_id') == img_id]
        
        word_boxes = []
        for ann in annotations_by_page:
            coords = _get_image_coords(ann)
            bbox = _polygon_to_bbox(coords)
            W, H = _get_image_size(ann, image_name_dict)
            bbox = _expand_bbox(bbox, W, H)
            text = _get_text_from_ann(ann)
            word_boxes.append({
                'bbox': bbox,
                'text': text.strip()
            })

        lines = group_words_into_lines(word_boxes,max_center_diff_factor=0.7)
        word_by_line = [[w['text'].strip() for w in line] for line in lines]
        bboxes = [merge_line_words(line) for line in lines]

        print( f"Imagen {images_name}: {len(word_boxes)} palabras agrupadas en {len(lines)} líneas."  )
        for bbox, words in zip(bboxes, word_by_line):
            transcript = re.sub(' +', ' ',' '.join(words) )
            if len(words) < 3 or len(words) > 15:
                continue
            x1, y1, x2, y2 = bbox
            crop = pil_img.crop((x1, y1, x2, y2))
            line_samples.append({"image": crop, 
                                "text": transcript })
            
    clean_samples, bad, stats = clean_lines_samples(line_samples,model_dir,cer_threshold=0.25)
    print(f"Líneas después de limpieza: {len(clean_samples)} (descartadas: {len(bad)})")
    print(f"Estadísticas de limpieza: {stats}")
    return clean_samples