# ============================================================
# 0. التثبيت وتحميل النموذج — تلقائيًا حسب توفر GPU أو لا
# ============================================================
import subprocess
import sys
import os

try:
    import segment_anything  # noqa: F401
except ImportError:
    print("تثبيت مكتبة segment-anything...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://github.com/facebookresearch/segment-anything.git"],
        check=True,
    )

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    MODEL_TYPE = "vit_l"
    CHECKPOINT = "sam_vit_l_0b3195.pth"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
else:
    MODEL_TYPE = "vit_b"
    CHECKPOINT = "sam_vit_b_01ec64.pth"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(CHECKPOINT):
    print(f"تحميل نموذج SAM ({MODEL_TYPE})...")
    import urllib.request
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)

# ============================================================
# 1. رفع الصورة
# ============================================================
def upload_image():
    try:
        from google.colab import files
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
    except ImportError:
        filename = "1.png"

    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"تعذر قراءة الصورة '{filename}'")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.title(f"الصورة المرفوعة: {filename}")
    plt.axis("off")
    plt.show()
    return img_rgb

# ============================================================
# 2. تحميل نموذج SAM وضبط إعدادات التوليد حسب الجهاز
# ============================================================
print(f"جاري التشغيل على: {device} (النموذج المستخدم: {MODEL_TYPE})")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32 if device == "cuda" else 16,
    pred_iou_thresh=0.7,
    stability_score_thresh=0.8,
    crop_n_layers=1 if device == "cuda" else 0,
    min_mask_region_area=30,
)

# ============================================================
# 3-7. توليد الأقنعة، الفلترة، الدمج، العرض، الحفظ
# ============================================================
def run_pipeline(img_rgb, output_dir="sam_symbols_out"):
    print("جاري توليد الأقنعة... انتظر ولا توقف التنفيذ حتى لو طوّل")
    masks = mask_generator.generate(img_rgb)
    print(f"عدد الأقنعة الخام التي أنتجها SAM: {len(masks)}")

    H, W = img_rgb.shape[:2]
    img_area = H * W
    MIN_AREA_RATIO = 0.0003
    MAX_AREA_RATIO = 0.5

    good_masks = [m for m in masks if MIN_AREA_RATIO * img_area <= m["area"] <= MAX_AREA_RATIO * img_area]

    def remove_contained(mask_list, containment_thresh=0.85):
        kept = []
        for m in sorted(mask_list, key=lambda x: x["area"], reverse=True):
            x, y, w, h = m["bbox"]
            contained = False
            for k in kept:
                kx, ky, kw, kh = k["bbox"]
                ix1, iy1 = max(x, kx), max(y, ky)
                ix2, iy2 = min(x + w, kx + kw), min(y + h, ky + kh)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                if (iw * ih) / (w * h + 1e-6) > containment_thresh:
                    contained = True
                    break
            if not contained:
                kept.append(m)
        return kept

    good_masks = remove_contained(good_masks)

    MAX_W_RATIO, MAX_H_RATIO = 0.3, 0.3
    before = len(good_masks)
    good_masks = [m for m in good_masks if m["bbox"][2] <= MAX_W_RATIO * W and m["bbox"][3] <= MAX_H_RATIO * H]
    if before != len(good_masks):
        print(f"تجاهلت {before - len(good_masks)} قناع كبير/ممتد بشكل غير طبيعي")

    before = len(good_masks)
    good_masks = [m for m in good_masks if int(m["bbox"][2]) > 0 and int(m["bbox"][3]) > 0]
    if before != len(good_masks):
        print(f"تجاهلت {before - len(good_masks)} قناع صندوقه صفري الحجم")

    print(f"عدد الأقنعة بعد الفلترة وإزالة المكرر: {len(good_masks)}")

    combined = np.zeros((H, W), dtype=np.uint8)
    for m in good_masks:
        combined[m["segmentation"]] = 1

    kernel = np.ones((3, 3), np.uint8)
    combined_eroded = cv2.erode(combined, kernel, iterations=1)
    num_labels, eroded_labels = cv2.connectedComponents(combined_eroded, connectivity=8)
    print(f"عدد المجموعات (رموز) بعد دمج المتلاصق: {num_labels - 1}")

    labels = np.zeros((H, W), dtype=np.int32)
    for m in good_masks:
        seg = m["segmentation"]
        votes = eroded_labels[seg]
        votes = votes[votes > 0]
        if votes.size == 0:
            continue
        vals, counts = np.unique(votes, return_counts=True)
        labels[seg] = vals[np.argmax(counts)]

    rng = np.random.default_rng(0)
    group_ids = [int(g) for g in np.unique(labels) if g != 0]
    group_colors = {gid: rng.random(3) for gid in group_ids}

    group_boxes = {}
    for gid in group_ids:
        ys, xs = np.where(labels == gid)
        group_boxes[gid] = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)

    def draw_grouped_overlay(ax, image_rgb, labels, group_colors, group_boxes):
        ax.imshow(image_rgb)
        overlay = np.zeros((*labels.shape, 4))
        for gid, color in group_colors.items():
            mask = labels == gid
            overlay[mask, 0] = color[0]
            overlay[mask, 1] = color[1]
            overlay[mask, 2] = color[2]
            overlay[mask, 3] = 0.45
        ax.imshow(overlay)
        for gid, color in group_colors.items():
            x0, x1, y0, y1 = group_boxes[gid]
            rect = plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)

    H_img, W_img = img_rgb.shape[:2]
    OVERVIEW_WIDTH_IN = 16
    overview_h = OVERVIEW_WIDTH_IN * (H_img / W_img)
    fig_top = plt.figure(figsize=(OVERVIEW_WIDTH_IN, overview_h))
    ax_top = fig_top.add_subplot(111)
    draw_grouped_overlay(ax_top, img_rgb, labels, group_colors, group_boxes)
    ax_top.set_title(f"{len(group_ids)} رمز بعد دمج المتلاصق (كل رمز بلون مستقل)")
    ax_top.axis("off")
    plt.tight_layout()
    plt.show()

    ZOOM_DISPLAY_SIZE = 320
    def zoom_for_display(image, target_size=ZOOM_DISPLAY_SIZE):
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return image
        scale = target_size / max(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        interp = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    CROP_COLS = 4
    crop_rows = int(np.ceil(len(group_ids) / CROP_COLS)) if group_ids else 1
    CELL_IN = 3.4

    fig_crops = plt.figure(figsize=(CROP_COLS * CELL_IN, crop_rows * CELL_IN))
    gs_crops = GridSpec(crop_rows, CROP_COLS, figure=fig_crops)

    for i, gid in enumerate(group_ids):
        x0, x1, y0, y1 = group_boxes[gid]
        crop = img_rgb[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        display_crop = zoom_for_display(crop)
        r, c = divmod(i, CROP_COLS)
        ax = fig_crops.add_subplot(gs_crops[r, c])
        ax.imshow(display_crop)
        ax.set_title(f"#{gid}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    os.makedirs(output_dir, exist_ok=True)
    for gid in group_ids:
        x0, x1, y0, y1 = group_boxes[gid]
        cv2.imwrite(f"{output_dir}/sign_{gid:03d}.png", cv2.cvtColor(img_rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))

    return group_ids, group_boxes
