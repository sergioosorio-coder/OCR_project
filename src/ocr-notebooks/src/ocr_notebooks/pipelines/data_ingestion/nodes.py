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

def _group_words_into_lines(word_boxes, y_tol_factor=0.6):
    for wb in word_boxes:
            x1, y1, x2, y2 = wb["bbox"]
            wb["y_center"] = (y1 + y2) / 2.0
            wb["height"] = (y2 - y1)
    word_boxes_sorted = sorted(word_boxes, key=lambda x: (x["y_center"]))

    lines = []
    current_line = []

    for wb in word_boxes_sorted:
        if not current_line:
            current_line.append(wb)
            continue

        ref = current_line[0]
        ref_center = ref["y_center"]
        ref_height = ref["height"]
        y_tol = ref_height * y_tol_factor

        if abs(wb["y_center"] - ref_center) <= y_tol:
            current_line.append(wb)
        else:
            lines.append(current_line)
            current_line = [wb]

    if current_line:
        lines.append(current_line)
    return lines

def merge_line_words(line_words):
    line_words = sorted(line_words, key=lambda w: w["bbox"][0])

    texts = [w["text"] for w in line_words]

    xs1 = [w["bbox"][0] for w in line_words]
    ys1 = [w["bbox"][1] for w in line_words]
    xs2 = [w["bbox"][2] for w in line_words]
    ys2 = [w["bbox"][3] for w in line_words]

    x1 = min(xs1)
    y1 = min(ys1)
    x2 = max(xs2)
    y2 = max(ys2)

    line_text = " ".join(texts)

    return (x1, y1, x2, y2), line_text



def extract_samples(
        coco_json_file:Dict[str, Any],
        images_zip_data,
        max_samples: int | None = 1000,
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
    for img_id in image_name_dict:
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

        lines = _group_words_into_lines(word_boxes,y_tol_factor=0.9)
        print( f"Imagen {images_name}: {len(word_boxes)} palabras agrupadas en {len(lines)} líneas." )
        
        for line_words in lines:
            line_bbox, line_text = merge_line_words(line_words)
            x1, y1, x2, y2 = line_bbox
            crop = pil_img.crop((x1, y1, x2, y2))
            
            line_samples.append({"image": crop, 
                                 "text": re.sub(' +', ' ', line_text.strip()) })
            
            if isinstance(max_samples,int) and len(line_samples) >= int(max_samples) :
                break
    return line_samples