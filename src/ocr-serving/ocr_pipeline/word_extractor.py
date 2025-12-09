import cv2
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass

def preprocess_line(line_bgr: np.ndarray, ksize=(5, 5), block_size=31, C=20) -> np.ndarray:
    gray = cv2.cvtColor(line_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, ksize, 0)
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    return thr

def remove_rulings(bin_line: np.ndarray) -> np.ndarray:
    h, w = bin_line.shape
    kernel_width = max(w // 10, 30)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    rulings = cv2.morphologyEx(bin_line, cv2.MORPH_OPEN, kernel)
    no_rulings = cv2.subtract(bin_line, rulings)
    no_rulings = cv2.morphologyEx(no_rulings, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return no_rulings

def get_text_components_from_bin(
    bin_img: np.ndarray,
    min_area: int = 60,
    min_height: int = 10,
    grow: bool = True
) -> List[Tuple[int, int, int, int]]:

    # 1) Opcional: agrandar un poco los trazos para unir trozos rotos
    cc_input = bin_img.copy()
    if grow:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        bin_big = cv2.dilate(cc_input, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cc_input, connectivity=8
    )

    components = []
    for label in range(1, num_labels):  # 0 es background
        x, y, w, h, area = stats[label]

        if area < min_area:
            continue
        if h < min_height:
            continue

        components.append((x, y, w, h))
    components.sort(key=lambda b: b[0])
    return components

def extract_word_crops(line_bgr: np.ndarray,
                       word_boxes: List[Tuple[int, int, int, int]]) -> List[np.ndarray]:
    """
    Devuelve lista de crops BGR de cada palabra.
    """
    crops = []
    for (x, y, w, h) in word_boxes:
        crop = line_bgr[y:y+h, x:x+w]
        crops.append(crop)
    return crops


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


def group_components_into_words(components):
    if not components:
        return []

    boxes = [Box(*c) for c in components]
    gaps = []

    for i in range(len(boxes) - 1):
        gap = boxes[i+1].x - (boxes[i].x + boxes[i].w)
        if gap > 0:
            gaps.append(gap)

    if not gaps:
        return [(b.x, b.y, b.w, b.h) for b in boxes]

    threshold = float(np.percentile(gaps, 75))
    word_boxes = []
    current = boxes[0]

    for b in boxes[1:]:
        gap = b.x - (current.x + current.w)
        if gap > threshold:
            word_boxes.append(current)
            current = b
        else:
            x1 = min(current.x, b.x)
            y1 = min(current.y, b.y)
            x2 = max(current.x + current.w, b.x + b.w)
            y2 = max(current.y + current.h, b.y + b.h)
            current = Box(x1, y1, x2 - x1, y2 - y1)

    word_boxes.append(current)
    return [(wb.x, wb.y, wb.w, wb.h) for wb in word_boxes]


def detect_words(line_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    bin_line = preprocess_line(line_bgr)
    bin_no_rulings = remove_rulings(bin_line)
    components = get_text_components_from_bin(
        bin_no_rulings,
        min_area=60, 
        min_height=10
    )
    word_boxes = group_components_into_words(components)

    outdir="data/crops"
    outdir = Path(outdir)
    outpath = outdir / f"crop_no_rulling.png"
    cv2.imwrite(str(outpath), bin_no_rulings)

    return word_boxes