"""تحويل بيانات CompVis إلى صيغة YOLO (كشف صنف واحد: أي رمز مسماري)."""
import os, glob, ast, random
import pandas as pd
from PIL import Image
from shutil import copyfile

PROJECT_DIR = "/content/drive/MyDrive/cuneiform_project"
DATASET_SRC = "cuneiform-sign-detection-dataset"
YOLO_DIR    = os.path.join(PROJECT_DIR, "yolo_dataset_singleclass")
IMAGES_DIR  = os.path.join(PROJECT_DIR, "yolo_dataset", "images_raw")

def main():
    for split in ["train", "val"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split), exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(DATASET_SRC, "annotations", "bbox_annotations_train_*.csv")))
    df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    df["bbox"] = df["bbox"].apply(ast.literal_eval)

    good_tablets = [t for t in df["tablet_CDLI"].unique() if os.path.exists(os.path.join(IMAGES_DIR, f"{t}.jpg"))]
    df = df[df["tablet_CDLI"].isin(good_tablets)].reset_index(drop=True)

    random.seed(42)
    random.shuffle(good_tablets)
    n_val = max(1, int(0.15 * len(good_tablets)))
    val_tablets = set(good_tablets[:n_val])

    size_cache = {}
    for cdli, group in df.groupby("tablet_CDLI"):
        split = "val" if cdli in val_tablets else "train"
        src_img = os.path.join(IMAGES_DIR, f"{cdli}.jpg")
        dst_img = os.path.join(YOLO_DIR, "images", split, f"{cdli}.jpg")
        if not os.path.exists(dst_img):
            copyfile(src_img, dst_img)

        if cdli not in size_cache:
            with Image.open(src_img) as im:
                size_cache[cdli] = im.size
        img_w, img_h = size_cache[cdli]

        lines = set()
        for _, row in group.iterrows():
            x1, y1, x2, y2 = row["bbox"]
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            xc = ((xmin + xmax) / 2) / img_w
            yc = ((ymin + ymax) / 2) / img_h
            w  = (xmax - xmin) / img_w
            h  = (ymax - ymin) / img_h
            if w <= 0 or h <= 0 or not (0 <= xc <= 1) or not (0 <= yc <= 1):
                continue
            lines.add(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        with open(os.path.join(YOLO_DIR, "labels", split, f"{cdli}.txt"), "w") as f:
            f.write("\n".join(lines))

    with open(os.path.join(YOLO_DIR, "data.yaml"), "w") as f:
        f.write(f"path: {YOLO_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['cuneiform_sign']\n")

    print("تم تجهيز بيانات YOLO بنجاح.")

if __name__ == "__main__":
    main()
