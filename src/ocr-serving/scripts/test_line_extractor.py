from ocr_pipeline.line_extractor import detect_lines
import cv2

path = "data/IMG_7330.jpg"
img = cv2.imread(path)
boxes, _, _, _ = detect_lines(img,debug=True)

draw = img.copy()
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite("data/debug_boxes.png", draw)
