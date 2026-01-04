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

# ✅ Your Google Drive model link (public share link)
DEFAULT_GDRIVE_LINK = "https://drive.google.com/file/d/1vMtdPhel-Bq1YhwArYqdPHoFJLEPmiqx/view?usp=sharing"


# =============================================================================
# Google Drive download helpers (for Streamlit Cloud)
# =============================================================================
def _gdrive_file_id(url_or_id: str) -> str:
    """Extract Google Drive file id from a share link OR accept an id directly."""
    s = (url_or_id or "").strip()

    # already looks like an id
    if re.fullmatch(r"[a-zA-Z0-9_-]{20,}", s):
        return s

    m = re.search(r"/file/d/([^/]+)", s)
    if m:
        return m.group(1)

    m = re.search(r"[?&]id=([^&]+)", s)
    if m:
        return m.group(1)

    raise ValueError("Could not extract Google Drive file id from the provided link.")


def _gdrive_download(file_id: str, dst_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Downloads a public Google Drive file to dst_path using confirm-token flow
    (works for large files too).
    """
    dst_path = str(dst_path)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)

    URL = "https://drive.google.com/uc?export=download"
    sess = requests.Session()

    # 1st request
    r = sess.get(URL, params={"id": file_id}, stream=True, timeout=60)
    r.raise_for_status()

    # handle Google Drive large-file confirm token
    token = None
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    if token:
        r = sess.get(URL, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
        r.raise_for_status()

    # write file
    with open(dst_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

    # sanity check: if too small or looks like html, it's probably an error page
    if os.path.getsize(dst_path) < 100 * 1024:  # <100KB
        with open(dst_path, "rb") as f:
            head = f.read(600).lower()
        if b"html" in head or b"google" in head:
            raise RuntimeError(
                "Downloaded file looks like an HTML page, not the model. "
                "Make sure your Google Drive link is PUBLIC: Anyone with the link -> Viewer."
            )

    return dst_path


def ensure_model_exists(local_path: str, gdrive_link_or_id: str) -> str:
    """
    If local model file doesn't exist, download it from Google Drive.
    Only applies to FILE paths (e.g. models/best_model.keras).
    """
    p = Path(local_path)

    # Only auto-download for file paths with typical model extensions
    if p.suffix.lower() in (".keras", ".h5"):
        if p.exists() and p.is_file():
            return str(p)

        file_id = _gdrive_file_id(gdrive_link_or_id)
        return _gdrive_download(file_id, str(p))

    # If it's a folder path (SavedModel), we don't auto-download here.
    return str(p)


# =============================================================================
# Model wrapper (Keras OR SavedModel)
# =============================================================================
class ModelRunner:
    """
    Unifies inference for:
    1) Keras model (has .predict)
    2) TF SavedModel user object (no .predict) -> uses serving_default signature
    """
    def __init__(self, obj, kind: str):
        self.obj = obj
        self.kind = kind

        if self.kind == "savedmodel":
            # Get callable signature
            self.fn = self.obj.signatures.get("serving_default", None)
            if self.fn is None:
                keys = list(self.obj.signatures.keys())
                if not keys:
                    raise ValueError("Loaded SavedModel has no signatures.")
                self.fn = self.obj.signatures[keys[0]]

            # Determine input key (commonly 'inputs' or first key)
            self.input_key = list(self.fn.structured_input_signature[1].keys())[0]

            # Determine output key (commonly 'output_0' or first key)
            self.output_key = list(self.fn.structured_outputs.keys())[0]

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        """
        x: (1,H,W,C) float32
        returns prob: (H,W) float32 in [0,1]
        """
        x = x.astype(np.float32)

        if self.kind == "keras":
            y = self.obj.predict(x, verbose=0)
            return y[0, ..., 0].astype(np.float32)

        xt = tf.convert_to_tensor(x, dtype=tf.float32)
        out = self.fn(**{self.input_key: xt})
        y = out[self.output_key].numpy()
        return y[0, ..., 0].astype(np.float32)


# =============================================================================
# Model loading (auto-download if missing)
# =============================================================================
def load_model_any(path: str) -> ModelRunner:
    """
    Supports:
      - Keras .h5/.keras => Keras model
      - SavedModel folder:
          * If it loads as Keras model, use keras
          * Else fallback to tf.saved_model.load() and use signatures

    Also: if path is a missing .keras/.h5 file, auto-download from Drive.
    """
    if not path:
        raise ValueError("Model path is empty.")

    # Allow overriding Drive link via environment (optional)
    gdrive_link = os.getenv("MODEL_GDRIVE_URL", DEFAULT_GDRIVE_LINK)

    # If user points to a .keras/.h5 file that doesn't exist, download it
    path = ensure_model_exists(path, gdrive_link)

    # 1) If it's a file, load with tf.keras (predict available)
    if os.path.isfile(path):
        model = tf.keras.models.load_model(path, compile=False)
        return ModelRunner(model, "keras")

    # 2) If it's a folder, try Keras load first
    if os.path.isdir(path):
        try:
            model = tf.keras.models.load_model(path, compile=False)
            if hasattr(model, "predict"):
                return ModelRunner(model, "keras")
        except Exception:
            pass

        # 3) Fall back to raw SavedModel load (signature-based)
        sm = tf.saved_model.load(path)
        return ModelRunner(sm, "savedmodel")

    raise ValueError(f"Model path does not exist: {path}")


# =============================================================================
# IO helpers
# =============================================================================
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


# =============================================================================
# Postprocessing
# =============================================================================
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


# =============================================================================
# Main prediction
# =============================================================================
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



