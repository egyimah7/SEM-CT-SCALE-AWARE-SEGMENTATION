# predict.py
import os
import numpy as np
import tensorflow as tf
from skimage.measure import label
from skimage.morphology import binary_closing, disk, remove_small_objects

from analysis import morphology_report, seg_metrics

IMAGE_SIZE = (512, 512)


# ---------------------------
# Model wrapper (Keras OR SavedModel)
# ---------------------------
class ModelRunner:
    """
    Unifies inference for:
    1) Keras model (has .predict)
    2) TF SavedModel user object (no .predict) -> uses signature
    """
    def __init__(self, obj, kind: str):
        self.obj = obj
        self.kind = kind

        if self.kind == "savedmodel":
            fn = self.obj.signatures.get("serving_default", None)
            if fn is None:
                keys = list(self.obj.signatures.keys())
                if not keys:
                    raise ValueError("Loaded SavedModel has no signatures.")
                fn = self.obj.signatures[keys[0]]
            self.fn = fn
            self.input_key = list(self.fn.structured_input_signature[1].keys())[0]
            self.output_key = list(self.fn.structured_outputs.keys())[0]

    def predict_prob(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)

        if self.kind == "keras":
            y = self.obj.predict(x, verbose=0)
            return y[0, ..., 0].astype(np.float32)

        xt = tf.convert_to_tensor(x, dtype=tf.float32)
        out = self.fn(**{self.input_key: xt})
        y = out[self.output_key].numpy()
        return y[0, ..., 0].astype(np.float32)


def load_model_any(path: str) -> ModelRunner:
    """
    Supports:
      - .keras/.h5 => Keras model
      - SavedModel directory => signature-based
    """
    if not path:
        raise ValueError("Model path is empty.")

    if os.path.isfile(path):
        model = tf.keras.models.load_model(path, compile=False)
        return ModelRunner(model, "keras")

    if os.path.isdir(path):
        # try Keras directory load
        try:
            model = tf.keras.models.load_model(path, compile=False)
            if hasattr(model, "predict"):
                return ModelRunner(model, "keras")
        except Exception:
            pass
        sm = tf.saved_model.load(path)
        return ModelRunner(sm, "savedmodel")

    raise ValueError(f"Model path does not exist: {path}")


# ---------------------------
# IO helpers
# ---------------------------
def read_image_bytes_to_float01(file_bytes: bytes) -> tf.Tensor:
    img = tf.io.decode_image(file_bytes, channels=1, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method="area")
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    return img  # HxWx1 float32


def read_mask_bytes_to_binary(file_bytes: bytes) -> tf.Tensor:
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
    s = np.clip(s, -3.0, 3.0) / 3.0  # normalize to [-1,1]
    return np.ones((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32) * np.float32(s)


# ---------------------------
# Postprocessing
# ---------------------------
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


# ---------------------------
# Main prediction
# ---------------------------
def predict_single(
    model_runner: ModelRunner,
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
    img01 = img_tf.numpy().squeeze()  # (H,W) float [0,1]

    scale_map = make_scale_map(res_um_px, canonical_res)
    x = np.concatenate([img01[..., None], scale_map], axis=-1)[None, ...]  # (1,H,W,2)

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






