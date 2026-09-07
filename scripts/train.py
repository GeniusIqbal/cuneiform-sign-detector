from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="/content/drive/MyDrive/cuneiform_project/yolo_dataset_singleclass/data.yaml",
        epochs=50,
        imgsz=960,
        project="/content/drive/MyDrive/cuneiform_project/runs",
        name="cuneiform_yolo_singleclass",
    )

if __name__ == "__main__":
    main()
