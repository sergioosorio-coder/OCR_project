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

def _crop_polygon_cv2(zf, file_path, coords):
    with zf.open(file_path) as f:
        pil_img = Image.open(f).convert("RGB")
    img = cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
    polygon_points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    pts = np.array(polygon_points, dtype=np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)  
    masked = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(pts)
    cropped = masked[y:y+h, x:x+w]
    return cropped

def extract_samples(
        coco_json_file:Dict[str, Any],
        images_zip_data,
        max_samples: int | None = 1000,
                    ):
    categories_dict = {cat['id']:cat['name'] for cat in coco_json_file['categories']}
    image_name_dict = {image['id']:image['file_name'] for image in coco_json_file['images']}
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

    samples = []
    for ann in annotations:
        if isinstance(max_samples,int) and len(samples) >= int(max_samples) :
            break

        text_gt = _get_text_from_ann(ann)
        if not text_gt or len(text_gt.strip()) < 3:
            continue
        
        images_name = image_name_dict[ann['image_id']]
        zip_member = name_map.get(images_name,'Empty')

        coords = _get_image_coords(ann)
        crop = _crop_polygon_cv2(zf,zip_member,coords)
        samples.append({
            'image':Image.fromarray(crop), 
            'text':text_gt.strip()
            })

    return samples