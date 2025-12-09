from ocr_pipeline.line_extractor import detect_lines
import cv2

# "data/IMG_7330.jpg"
# "data/en_hw2022_01_1.jpg"
# "data/en_hw2022_07_IMG_20211001_135020.jpg"

path = "data/en_hw2022_07_IMG_20211001_135020.jpg"
img = cv2.imread(path)
boxes, _, _, _ = detect_lines(img,debug=True,debug_path = './data/debugs/',)

draw = img.copy()
for (x1, y1, x2, y2) in boxes:
    cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imwrite("data/debugs/debug_boxes.png", draw)

