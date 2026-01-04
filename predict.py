# predict.py
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import requests

from skimage.measure import label
from skimage.morphology import binary_closing, disk, remove_small_objects

from analysis import morphology_report, seg_metrics

IMAGE_SIZE = (512, 512)

# ✅ Your Google Drive FILE link (works)
MODEL_URL = "https://drive.google.com/file/d/1vMtdPhel-Bq1YhwArYqdPHoFJLEPmiqx/view?usp=sharing"
LOCAL_MODEL_PATH = "models/best_model.keras"


# ---------------------------------------------------------------------
# Google Drive download helper (handles confirm token)
# ---------------------------------------------------------------------
def _gdrive_file_id(url: str) -> str:
    m = re.search(r"/file/d/([^/]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        return m.group(1)
    raise ValueError("Could not extract Google Drive file id from URL.")


def _download_gdrive(file_url: str, dst_path: str) -> str:
    file_id = _gdrive_file_id(file_url)
    dst = Path(dst_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    base = "https://drive.google.com/uc?export=download"
    sess = requests.Session()

    r = sess.get(base, params={"id": file_id}, stream=True, timeout=120)
    r.raise_for_status()

    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        r = sess.get(base, params={"id": file_id, "confirm": token}, stream=True, timeout=120)
        r.raise_for_status()

    with open(dst, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    # Sanity check
    if dst.stat().st_size < 200 * 1024:
        with open(dst, "rb") as f:
            head = f.read(600).lower()
        if b"html" in head:
            raise RuntimeError(
                "Downloaded HTML instead of model. "
                "Make sure the Drive file is shared as: Anyone with the link (Viewer)."
            )

    return str(dst)


def ensure_model_file(local_path: str = LOCAL_MODEL_PATH, url: str = MODEL_URL) -> str:
    p = Path(local_path)
    if p.exists() and p.stat().st_size > 0:
        return str(p)
    return _download_gdrive(url, str(p))


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------
class ModelRunner:
    def __init__(self, model):
        self.model = model

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        y = self.model.predict(x.astype(np.float32), verbose=0)  # (1,H,W,1)
        return y[0, ..., 0].astype(np.float32)


def load_model_any(path: str) -> ModelRunner:
    """
    If the model file doesn't exist locally, download it from Google Drive.
    """
    # If user typed models/best_model.keras, we download it into that same path
    if path.endswith(".keras") and not os.path.exists(path):
        ensure_model_file(path, MODEL_URL)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. "
            f"Expected it to be downloaded to this path."
        )

    model = tf.keras.models.load_model(path, compile=False)
    return ModelRunner(model)


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def read_image_bytes_to_float01(file_bytes: bytes):
    img = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method="area")
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img  # HxWx1 float32 [0,1]


def read_mask_bytes_to_binary(file_bytes: bytes):
    """
    GT convention: crack/pore is BLACK in the image.
    Converts to 1 for crack/pore, 0 for matrix.
    """
    msk = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)
    msk = tf.image.resize(msk, IMAGE_SIZE, method="nearest")
    msk = tf.cast(msk < 128, tf.uint8)  # black->1
    return msk


def make_scale_map(res_um_px: float, canonical_res: float) -> np.ndarray:
    s = np.log((float(res_um_px) / float(canonical_res)) + 1e-12)
    s = np.clip(s, -3.0, 3.0) / 3.0
    return np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32) * np.float32(s)


# ---------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------
def hysteresis_mask(prob: np.ndarray, t_low=0.35, t_high=0.70, min_size=10, close_radius=1) -> np.ndarray:
    seeds = prob >= t_high
    cand = prob >= t_low
    lab = label(cand, connectivity=2)

    keep = np.zeros_like(cand, dtype=bool)
    seed_labels = np.unique(lab[seeds])
    seed_labels = seed_labels[seed_labels != 0]

    for sl in seed_labels:
        keep |= (lab == sl)

    if close_radius and close_radius > 0:
        keep = binary_closing(keep, footprint=disk(int(close_radius)))

    if min_size and min_size > 0:
        keep = remove_small_objects(keep, min_size=int(min_size))

    return keep.astype(np.uint8)


# ---------------------------------------------------------------------
# Main prediction
# ---------------------------------------------------------------------
def predict_single(
    model_runner: ModelRunner,
    image_bytes: bytes,
    res_um_px: float,
    canonical_res: float = 2.65,
    t_low: float = 0.35,
    t_high: float = 0.70,
    min_obj_px: int = 10,
    close_radius: int = 1,
    gt_mask_bytes: Optional[bytes] = None,
):
    img_tf = read_image_bytes_to_float01(image_bytes)
    img01 = img_tf.numpy().squeeze()  # (H,W)

    scale_map = make_scale_map(res_um_px, canonical_res)
    x = np.concatenate([img01[..., None], scale_map], axis=-1)[None, ...]  # (1,512,512,2)

    prob = model_runner.predict_prob(x)
    pred = hysteresis_mask(prob, t_low=t_low, t_high=t_high, min_size=min_obj_px, close_radius=close_radius)

    metrics = None
    if gt_mask_bytes:
        gt = read_mask_bytes_to_binary(gt_mask_bytes).numpy().squeeze().astype(np.uint8)
        metrics = seg_metrics(gt, pred)

    morph_summary, df_objects = morphology_report(pred, float(res_um_px), min_obj_px=0)

    return {
        "img01": img01,
        "prob": prob,
        "pred": pred,
        "metrics": metrics,
        "morph_summary": morph_summary,
        "df_objects": df_objects,
    }





