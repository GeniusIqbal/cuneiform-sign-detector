from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
import glob
import matplotlib.pyplot as plt

WEIGHTS = "/content/drive/MyDrive/cuneiform_project/runs/cuneiform_yolo_singleclass/weights/best.pt"
LABEL_AR = "رمز مسماري"

def _get_arabic_font(size=20):
    candidates = (
        glob.glob("/usr/share/fonts/**/NotoNaskhArabic-Regular.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/NotoSansArabic-Regular.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/*Arabic*.ttf", recursive=True)
    )
    if candidates:
        return ImageFont.truetype(candidates[0], size)
    return ImageFont.load_default()

def _arabic_text(text):
    return get_display(arabic_reshaper.reshape(text))

def predict_image(image_path, weights=WEIGHTS, conf=0.15, iou=0.3, imgsz=960):
    model = YOLO(weights)
    results = model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou, save=False)
    r = results[0]

    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_arabic_font(20)
    label_text = _arabic_text(LABEL_AR)

    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(boxes, confs):
        draw.rectangle([x1, y1, x2, y2], outline=(0, 120, 255), width=2)
        text = f"{label_text} {c:.2f}"
        tw = draw.textlength(text, font=font)
        draw.rectangle([x1, max(0, y1 - 22), x1 + tw + 6, y1], fill=(0, 120, 255))
        draw.text((x1 + 3, max(0, y1 - 22)), text, font=font, fill=(255, 255, 255))

    plt.figure(figsize=(14, 14))
    plt.imshow(pil_img)
    plt.axis("off")
    plt.title(f"عدد الرموز المكتشفة: {len(boxes)}")
    plt.show()
    return results
