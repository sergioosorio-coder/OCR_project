import cv2
import numpy as np
from pathlib import Path

def to_gray_and_blur(img_bgr, ksize=(5, 5)):
    """
    Convierte la imagen BGR a escala de grises y aplica un suavizado Gaussiano.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, ksize, 0)
    return gray

def binarize_page(gray, block_size=31, C=20):
    """
    Binarización adaptativa (texto + líneas del cuaderno en blanco = 255).
    Se devuelve la versión invertida (texto en 255 sobre fondo 0).
    """
    thr = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C
    )
    return thr

def detect_rulings(thr, kernel_width=120):
    """
    Detecta las líneas horizontales largas (renglones) usando morfología.
    kernel_width define cuán largas deben ser las líneas detectadas.
    """
    kernel_rulings = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
    rulings = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel_rulings, iterations=1)
    return rulings

def remove_rulings(thr, rulings, clean_kernel=(3, 3)):
    """
    Elimina las líneas detectadas del binario y limpia ruido pequeño.
    """
    thr_no_rulings = cv2.bitwise_and(thr, cv2.bitwise_not(rulings))

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, clean_kernel)
    thr_no_rulings = cv2.morphologyEx(thr_no_rulings, cv2.MORPH_OPEN, kernel_small, iterations=1)
    return thr_no_rulings

def connect_text_horizontally(thr_no_rulings, kernel_size=(40, 3)):
    """
    Une caracteres de una misma línea usando dilatación horizontal.
    Esto genera bandas continuas donde hay texto.
    """
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    band_mask = cv2.dilate(thr_no_rulings, kernel_h, iterations=1)
    return band_mask

def refine_vertical_bounds(thr_no_rulings,
                           y0,
                           y1,
                           extra_ratio=0.5,
                           min_extra=15):
    """
    Refina los límites verticales (y0, y1) buscando exactamente
    dónde hay píxeles de texto en thr_no_rulings, y luego agrega
    un padding proporcional a la altura de la línea.

    extra_ratio: % de la altura de la línea que se añade por arriba y abajo.
    min_extra:  padding mínimo en píxeles.
    """
    h_total = thr_no_rulings.shape[0]

    # Recorte de la banda original
    band = thr_no_rulings[y0:y1, :]
    row_hist = np.sum(band > 0, axis=1)
    idx = np.where(row_hist > 0)[0]

    # Si no hay texto, devolvemos lo original
    if len(idx) == 0:
        return y0, y1

    # Donde realmente empiezan / terminan las letras
    top = y0 + int(idx[0])
    bottom = y0 + int(idx[-1])

    height = bottom - top + 1
    extra = max(int(height * extra_ratio), min_extra)

    y0_new = max(top - extra, 0)
    y1_new = min(bottom + extra, h_total - 1)

    return y0_new, y1_new

def get_line_spans_from_projection(band_mask, min_line_height=15):
    """
    A partir de la proyección horizontal (conteo de píxeles por fila),
    devuelve una lista de segmentos verticales (y0, y1) que contienen texto.
    """
    hist = np.sum(band_mask > 0, axis=1)

    # Umbral dinámico en función de la media y la desviación estándar
    row_mean = np.mean(hist)
    row_std = np.std(hist)
    row_thresh = row_mean + 0.5 * row_std

    is_text_row = hist > row_thresh

    line_spans = []
    in_run = False
    start = 0

    for y, flag in enumerate(is_text_row):
        if flag and not in_run:
            start = y
            in_run = True
        elif not flag and in_run:
            end = y
            if end - start >= min_line_height:
                line_spans.append((start, end))
            in_run = False

    # Si la imagen termina dentro de una banda
    if in_run:
        end = len(is_text_row)
        if end - start >= min_line_height:
            line_spans.append((start, end))

    return line_spans

def build_line_boxes(thr_no_rulings, line_spans,
                     density_thresh=0.02,
                     pad_x=5,
                     extra_y_ratio=0.5,
                     min_extra_y=15):
    """
    Para cada span vertical (y0, y1):
      - se refina el rango en Y usando thr_no_rulings
      - se calcula densidad de texto
      - se construye el bounding box (x0, y0_ref, x1, y1_ref) con padding en X
    """
    h, w = thr_no_rulings.shape
    boxes = []

    for (y0, y1) in line_spans:
        # 1) Refinar verticalmente en función del texto real
        y0_ref, y1_ref = refine_vertical_bounds(
            thr_no_rulings,
            y0,
            y1,
            extra_ratio=extra_y_ratio,
            min_extra=min_extra_y
        )

        # 2) Densidad de texto dentro del rango refinado
        band = thr_no_rulings[y0_ref:y1_ref, :]
        text_pixels = np.sum(band > 0)
        area = (y1_ref - y0_ref) * w
        density = text_pixels / float(area + 1e-6)
        if density < density_thresh:
            continue

        # 3) Bounding box horizontal
        col_hist = np.sum(band > 0, axis=0)
        cols = np.where(col_hist > 0)[0]
        if len(cols) == 0:
            continue

        x0 = int(cols[0])
        x1 = int(cols[-1])

        # Padding en X
        x0 = max(x0 - pad_x, 0)
        x1 = min(x1 + pad_x, w - 1)

        boxes.append((x0, y0_ref, x1, y1_ref))

    return boxes

def detect_lines(img_bgr,
                min_line_height=15,
                density_thresh=0.02,
                debug_path = './data/',
                debug=False):
    
    gray = to_gray_and_blur(img_bgr)
    thr = binarize_page(gray)
    rulings = detect_rulings(thr)
    thr_no_rulings = remove_rulings(thr, rulings)
    band_mask = connect_text_horizontally(thr_no_rulings)
    line_spans = get_line_spans_from_projection(
            band_mask, min_line_height=min_line_height
        )
    boxes = build_line_boxes(
        thr_no_rulings,
        line_spans,
        density_thresh=density_thresh
    )   
    if debug:
        outdir = Path(debug_path)
        outdir.mkdir(parents=True, exist_ok=True)

        print("line_spans:", line_spans)
        print("boxes:", boxes)
        cv2.imwrite(debug_path+"debug_1_gray.png", gray)
        cv2.imwrite(debug_path+"debug_2_thr.png", thr)
        cv2.imwrite(debug_path+"debug_3_rulings.png", rulings)
        cv2.imwrite(debug_path+"debug_4_thr_no_rulings.png", thr_no_rulings)
        cv2.imwrite(debug_path+"debug_5_band_mask.png", band_mask)

    return boxes, thr, thr_no_rulings, band_mask