import subprocess
import sys
import os

PROJECT_DIR = "/content/drive/MyDrive/cuneiform_project"
os.makedirs(f"{PROJECT_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{PROJECT_DIR}/sam_symbols_out", exist_ok=True)

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
    CHECKPOINT = f"{PROJECT_DIR}/checkpoints/sam_vit_l_0b3195.pth"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
else:
    MODEL_TYPE = "vit_b"
    CHECKPOINT = f"{PROJECT_DIR}/checkpoints/sam_vit_b_01ec64.pth"
    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

if not os.path.exists(CHECKPOINT):
    print(f"تحميل نموذج SAM ({MODEL_TYPE})... (أول مرة بس، بعدها يصير محفوظ بدرايف للأبد)")
    import urllib.request
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT)
else:
    print(f"وجدت نموذج SAM محفوظ مسبقًا بدرايف، ما راح يتحمل من جديد")

# (باقي الكود نفسه بدون تغيير: رفع الصورة، تحميل sam، run_pipeline...
#  فقط استبدلي output_dir الافتراضي بهذا:)
DEFAULT_OUTPUT_DIR = f"{PROJECT_DIR}/sam_symbols_out"
