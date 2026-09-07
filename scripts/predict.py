from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

WEIGHTS = "/content/drive/MyDrive/cuneiform_project/runs/cuneiform_yolo_singleclass/weights/best.pt"

def predict_image(image_path, weights=WEIGHTS, conf=0.15, iou=0.3, imgsz=960):
    model = YOLO(weights)
    results = model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou, save=False)
    result_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(14, 14))
    plt.imshow(result_img)
    plt.axis("off")
    plt.title(f"عدد الرموز المكتشفة: {len(results[0].boxes)}")
    plt.show()
    return results
