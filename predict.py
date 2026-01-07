# predict.py (Google Drive download version)
import os
import numpy as np
import tensorflow as tf
import gdown
from skimage.measure import label
from skimage.morphology import binary_closing, disk, remove_small_objects

from analysis import morphology_report, seg_metrics

IMAGE_SIZE = (512, 512)

# Your Google Drive FILE link
GDRIVE_URL = "https://drive.google.com/file/d/1vMtdPhel-Bq1YhwArYqdPHoFJLEPmiqx/view?usp=sharing"
DEFAULT_MODEL_LOCAL = "models/best_model.keras"


def ensure_model_downloaded(dst_path: str = DEFAULT_MODEL_LOCAL) -> str:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 10_000_000:
        return dst_path

    # gdown can download from a file URL directly
    gdown.download(url=GDRIVE_URL, output=dst_path, quiet=False, fuzzy=True)

    if not os.path.exists(dst_path):
        raise FileNotFoundError(f"Download failed, model not found at {dst_path}")
    return dst_path


class ModelRunner:
    def __init__(self, model):
        self.model = model

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        y = self.model.predict(x.astype(np.float32), verbose=0)
        return y[0, ..., 0].astype(np.float32)


def load_model_any(path: str | None = None) -> ModelRunner:
    if not path:
        path = DEFAULT_MODEL_LOCAL

    if path == DEFAULT_MODEL_LOCAL:
        path = ensure_model_downloaded(path)

    model = tf.keras.models.load_model(path, compile=False)
    return ModelRunner(model)


def read_image_bytes_to_float01(file_bytes: bytes):
    img = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method="area")
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def read_mask_bytes_to_binary(file_bytes: bytes):
    msk = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)
    msk = tf.image.resize(msk, IMAGE_SIZE, method="nearest")
    msk = tf.cast(msk < 128, tf.uint8)
    return msk


def make_scale_map(res_um_px: float, canonical_res: float) -> np.ndarray:
    s = np.log((float(res_um_px) / float(canonical_res)) + 1e-12)
    s = np.clip(s, -3.0, 3.0) / 3.0
    return np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32) * np.float32(s)


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


def predict_single(
    model_runner,
    image_bytes: bytes,
    res_um_px: float,
    canonical_res: float = 2.65,
    t_low: float = 0.35,
    t_high: float = 0.70,
    min_obj_px: int = 10,
    close_radius: int = 1,
    gt_mask_bytes: bytes | None = None,
):
    img_tf = read_image_bytes_to_float01(image_bytes)
    img01 = img_tf.numpy().squeeze()

    scale_map = make_scale_map(res_um_px, canonical_res)
    x = np.concatenate([img01[..., None], scale_map], axis=-1)[None, ...]

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







