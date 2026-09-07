# Cuneiform Sign Detector

كاشف رموز مسمارية باستخدام YOLOv8، مبني على بيانات CompVis (CDLI).

- `scripts/prepare_dataset.py` — تحويل بيانات CompVis إلى صيغة YOLO (صنف واحد).
- `scripts/train.py` — تدريب YOLOv8 على البيانات المجهزة.
- `scripts/predict.py` — تشغيل النموذج المدرب على صورة جديدة.

النموذج المدرب الحالي: mAP50 = 0.21 (38 صورة تدريب، 3300 صندوق).
